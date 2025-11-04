from __future__ import annotations

from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse
from django.http import HttpRequest, HttpResponse
from django.contrib import messages
from django.conf import settings

from .forms import DatasetUploadForm, RunConfigForm
from .models import Dataset, Run

import pandas as pd
import os


def home(request: HttpRequest) -> HttpResponse:
    datasets = Dataset.objects.order_by('-uploaded_at')[:10]
    return render(request, 'fs/home.html', {'datasets': datasets})


def upload_dataset(request: HttpRequest) -> HttpResponse:
    if request.method == 'POST':
        form = DatasetUploadForm(request.POST, request.FILES)
        if form.is_valid():
            ds: Dataset = form.save()
            try:
                df = pd.read_csv(ds.file.path, nrows=1)
                ds.columns = list(df.columns)
                ds.save(update_fields=['columns'])
                messages.success(request, 'تم رفع البيانات بنجاح. انتقل للإعداد والتشغيل.')
                return redirect('configure_run', dataset_id=ds.id)
            except Exception as e:
                ds.delete()
                messages.error(request, f'تعذّر قراءة ملف CSV: {e}')
    else:
        form = DatasetUploadForm()
    return render(request, 'fs/upload.html', {'form': form})


def configure_run(request: HttpRequest, dataset_id: int) -> HttpResponse:
    ds = get_object_or_404(Dataset, pk=dataset_id)

    if not ds.columns:
        try:
            df = pd.read_csv(ds.file.path, nrows=1)
            ds.columns = list(df.columns)
            ds.save(update_fields=['columns'])
        except Exception as e:
            messages.error(request, f'تعذّر قراءة أعمدة الملف: {e}')
            return redirect('upload_dataset')

    if request.method == 'POST':
        form = RunConfigForm(request.POST, columns=ds.columns)
        if form.is_valid():
            target = form.cleaned_data['target_column']
            task = form.cleaned_data['task']
            estimator = form.cleaned_data['estimator']
            cv_folds = form.cleaned_data['cv_folds']
            quick_mode = form.cleaned_data.get('quick_mode') or False
            params = {
                'pop_size': form.cleaned_data['pop_size'],
                'generations': form.cleaned_data['generations'],
                'crossover_rate': form.cleaned_data['crossover_rate'],
                'mutation_rate': form.cleaned_data['mutation_rate'],
                'elitism': form.cleaned_data['elitism'],
                'penalty': form.cleaned_data['penalty'],
            }

            # Apply quick-mode overrides for speed
            if quick_mode:
                cv_folds = min(cv_folds, 3)
                params['pop_size'] = min(int(params['pop_size']), 20)
                params['generations'] = min(int(params['generations']), 8)

            run = Run.objects.create(
                dataset=ds,
                target_column=target,
                task=task,
                estimator=estimator,
                cv_folds=cv_folds,
                ga_params=params,
            )

            # Execute GA and baselines (synchronously for MVP)
            try:
                from .utils import load_dataset, infer_task_type
                X, y, feature_names = load_dataset(ds.file.path, target)
                if task not in ('classification', 'regression'):
                    task = infer_task_type(y)

                # If quick mode and dataset is large, sample rows for speed
                if quick_mode and len(X) > 2000:
                    import numpy as np
                    sample_n = 2000
                    rng = np.random.default_rng(42)
                    idx = rng.choice(len(X), size=sample_n, replace=False)
                    X = X.iloc[idx]
                    y = y.iloc[idx]

                from .ga import run_ga
                ga_result = run_ga(X, y, task, estimator, cv_folds, params, feature_names)

                from .baselines import compute_baselines
                baseline_results = compute_baselines(X, y, task, estimator, cv_folds, feature_names, fast=quick_mode)

                run.best_score = ga_result.get('best_score', 0.0)
                run.selected_features = ga_result.get('selected_features', [])
                run.metrics = {
                    'ga': ga_result,
                }
                run.baselines = baseline_results
                run.task = task
                run.save()

                return redirect('run_detail', run_id=run.id)
            except Exception as e:
                messages.error(request, f'تعذّر تنفيذ الخوارزمية: {e}')
                run.delete()
    else:
        form = RunConfigForm(columns=ds.columns, initial={
            'task': 'classification',
            'estimator': 'logreg',
            'cv_folds': 5,
        })

    return render(request, 'fs/configure.html', {'dataset': ds, 'form': form})


def run_detail(request: HttpRequest, run_id: int) -> HttpResponse:
    run = get_object_or_404(Run, pk=run_id)
    return render(request, 'fs/results.html', {
        'run': run,
        'selected_features': run.selected_features or [],
        'metrics': run.metrics or {},
        'baselines': run.baselines or {},
    })

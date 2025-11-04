from django.db import models


class Dataset(models.Model):
    name = models.CharField(max_length=255)
    file = models.FileField(upload_to='datasets/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    columns = models.JSONField(default=list, blank=True)
    task_type = models.CharField(max_length=32, blank=True, default='')  # classification | regression

    def __str__(self):
        return self.name


class Run(models.Model):
    TASK_CHOICES = (
        ('classification', 'Classification'),
        ('regression', 'Regression'),
    )

    ESTIMATOR_CHOICES = (
        ('logreg', 'Logistic Regression / Linear Regression'),
        ('svm', 'SVM (Linear/LinearSVR)'),
        ('rf', 'Random Forest (Cls/Reg)'),
    )

    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='runs')
    target_column = models.CharField(max_length=255)
    task = models.CharField(max_length=32, choices=TASK_CHOICES)
    estimator = models.CharField(max_length=32, choices=ESTIMATOR_CHOICES, default='logreg')
    cv_folds = models.PositiveIntegerField(default=5)

    ga_params = models.JSONField(default=dict, blank=True)
    best_score = models.FloatField(default=0.0)
    selected_features = models.JSONField(default=list, blank=True)
    metrics = models.JSONField(default=dict, blank=True)
    baselines = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Run #{self.id} on {self.dataset.name}"


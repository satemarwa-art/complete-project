from django import forms
from .models import Dataset, Run


class DatasetUploadForm(forms.ModelForm):
    class Meta:
        model = Dataset
        fields = ['name', 'file']


class RunConfigForm(forms.Form):
    target_column = forms.ChoiceField(label='Target Column')
    task = forms.ChoiceField(choices=Run.TASK_CHOICES)
    estimator = forms.ChoiceField(choices=Run.ESTIMATOR_CHOICES)
    cv_folds = forms.IntegerField(min_value=2, max_value=20, initial=5)

    pop_size = forms.IntegerField(min_value=5, max_value=1000, initial=40)
    generations = forms.IntegerField(min_value=1, max_value=500, initial=20)
    crossover_rate = forms.FloatField(min_value=0.0, max_value=1.0, initial=0.8)
    mutation_rate = forms.FloatField(min_value=0.0, max_value=1.0, initial=0.05)
    elitism = forms.IntegerField(min_value=0, max_value=10, initial=2)
    penalty = forms.FloatField(min_value=0.0, max_value=1.0, initial=0.01, help_text='Penalty per feature proportion')
    quick_mode = forms.BooleanField(required=False, initial=True, label='الوضع السريع (أسرع تشغيل)')

    def __init__(self, *args, **kwargs):
        columns = kwargs.pop('columns', None)
        super().__init__(*args, **kwargs)
        if columns:
            self.fields['target_column'].choices = [(c, c) for c in columns]

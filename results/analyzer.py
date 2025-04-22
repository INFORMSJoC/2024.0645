# results/analyzer.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
from typing import List, Dict
from pathlib import Path

class ResultsAnalyzer:
    """��ά��ʵ��������"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.df = self._load_all_results()
        
    def _load_all_results(self) -> pd.DataFrame:
        """��������ʵ������DataFrame"""
        files = list(self.results_dir.glob("*.parquet"))
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    
    def compare_metrics(self, metric: str, 
                       group_by: str = "experiment_group") -> Dict:
        """ִ��ͳ�ƱȽϷ���"""
        groups = self.df[group_by].unique()
        results = {}
        
        # ANOVA����
        groups_data = [self.df[self.df[group_by]==g][metric] for g in groups]
        anova_result = stats.f_oneway(*groups_data)
        
        # �º���رȽ�
        posthoc = []
        for i in range(len(groups)):
            for j in range(i+1, len(groups)):
                t_test = stats.ttest_ind(groups_data[i], groups_data[j])
                posthoc.append({
                    'group1': groups[i],
                    'group2': groups[j],
                    'pvalue': t_test.pvalue
                })
        
        # У��pֵ
        pvals = [x['pvalue'] for x in posthoc]
        reject, pvals_corrected, _, _ = multipletests(pvals, method='fdr_bh')
        
        for i, item in enumerate(posthoc):
            item['pvalue_corrected'] = pvals_corrected[i]
            item['significant'] = reject[i]
        
        return {
            'anova': {'statistic': anova_result.statistic,
                     'pvalue': anova_result.pvalue},
            'posthoc': posthoc
        }
    
    def visualize_results(self, x: str, y: str, 
                         hue: str = None, save_path: str = None):
        """���ɽ���ʽ���ӻ�ͼ��"""
        plt.figure(figsize=(12, 8))
        
        if hue:
            sns.boxplot(x=x, y=y, hue=hue, data=self.df)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            sns.lineplot(x=x, y=y, data=self.df)
            
        plt.xticks(rotation=45)
        plt.title(f"{y} by {x}")
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def generate_report(self, output_path: str, 
                       metrics: List[str] = None):
        """����������������"""
        analysis = {}
        
        # ����ͳ����
        analysis['descriptive'] = self.df.describe().to_dict()
        
        # ��ָ��Ƚ�
        analysis['comparisons'] = {}
        metrics = metrics or self.df.select_dtypes(include=np.number).columns
        for metric in metrics:
            analysis['comparisons'][metric] = self.compare_metrics(metric)
        
        # ���ӻ�ͼ��
        vis_dir = Path(output_path).parent / "figures"
        vis_dir.mkdir(exist_ok=True)
        
        for metric in metrics:
            self.visualize_results(x='experiment_group', y=metric,
                                  save_path=vis_dir/f"{metric}_comparison.png")
        
        # ���汨��
        report = {
            'analysis': analysis,
            'visualizations': [str(p) for p in vis_dir.glob("*.png")]
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(report, f)
        
        return report

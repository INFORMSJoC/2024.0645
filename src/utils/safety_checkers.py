# src/utils/safety_checkers.py
import os
import re
import ast
import bandit
import safety
import cryptography
from typing import Dict, List, Tuple

class SafetyChecker:
    """��ȫ����������"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.banned_patterns = self.config.get('banned_patterns', [])
        self.allowed_libraries = self.config.get('allowed_libraries', [])
        self.security_threshold = self.config.get('security_threshold', 0.7)

    def check_code(self, code: str) -> Tuple[bool, List[str]]:
        """�����밲ȫ��"""
        issues = []
        
        # ����ֹ��ģʽ
        for pattern in self.banned_patterns:
            if re.search(pattern, code):
                issues.append(f"Detected banned pattern: {pattern}")
        
        # ʹ�� Bandit ���о�̬����
        try:
            bandit_result = bandit.main.bandi(code)
            if bandit_result.issue_count > 0:
                issues.extend([f"Bandit issue: {issue}" for issue in bandit_result.issues])
        except Exception as e:
            issues.append(f"Bandit error: {str(e)}")
        
        # ��������ȫ��
        try:
            safety_result = safety.check(code)
            if safety_result.vulnerabilities:
                issues.extend([f"Security vulnerability: {vuln}" for vuln in safety_result.vulnerabilities])
        except Exception as e:
            issues.append(f"Safety check error: {str(e)}")
        
        return len(issues) == 0, issues

    def check_dependencies(self, requirements: List[str]) -> Tuple[bool, List[str]]:
        """��������ȫ��"""
        issues = []
        
        # ����ֹ�Ŀ�
        for req in requirements:
            lib = req.split('==')[0]
            if lib not in self.allowed_libraries:
                issues.append(f"Unauthorized library: {lib}")
        
        # ʹ�� Safety ���©��
        try:
            safety_result = safety.check(requirements)
            if safety_result.vulnerabilities:
                issues.extend([f"Security vulnerability: {vuln}" for vuln in safety_result.vulnerabilities])
        except Exception as e:
            issues.append(f"Safety check error: {str(e)}")
        
        return len(issues) == 0, issues

    def check_model(self, model_path: str) -> Tuple[bool, List[str]]:
        """���ģ�Ͱ�ȫ��"""
        issues = []
        
        # ���ģ���ļ�
        if not os.path.exists(model_path):
            issues.append(f"Model file not found: {model_path}")
            return False, issues
        
        # ʹ�� Cryptography ���ģ��ǩ��
        try:
            with open(model_path, 'rb') as f:
                model_data = f.read()
            if not cryptography.verify_signature(model_data):
                issues.append("Model signature verification failed")
        except Exception as e:
            issues.append(f"Model security check error: {str(e)}")
        
        return len(issues) == 0, issues

# Contribution Guide

Merci de votre intérêt pour contribuer à ce projet ! 🎉

## Comment Contribuer

### 1. Fork le Repository
Cliquez sur le bouton "Fork" en haut à droite de la page GitHub.

### 2. Cloner votre Fork
```bash
git clone https://github.com/votre-username/Analyse-quantitative-d-une-action-boursi-re.git
cd Analyse-quantitative-d-une-action-boursi-re
```

### 3. Créer une Branche de Feature
```bash
git checkout -b feature/ma-nouvelle-feature
```

### 4. Faire vos Modifications
- Respectez le style de code PEP 8
- Ajoutez des docstrings à vos fonctions
- Écrivez des tests unitaires pour les nouvelles fonctionnalités

### 5. Commit et Push
```bash
git add .
git commit -m "Ajouter ma nouvelle feature"
git push origin feature/ma-nouvelle-feature
```

### 6. Créer une Pull Request
- Allez sur GitHub et cliquez sur "New Pull Request"
- Sélectionnez votre branche
- Remplissez le formulaire de PR avec une description claire

## Standards de Code

### Style de Code
- Respectez [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Utilisez `black` pour le formatage automatique
- Utilisez `flake8` pour la vérification de style

```bash
# Formater le code
black src/

# Vérifier le style
flake8 src/
```

### Docstrings
Utilisez le format NumPy pour les docstrings :

```python
def ma_fonction(param1: int, param2: str) -> bool:
    """
    Description courte de la fonction.
    
    Description plus longue si nécessaire.
    
    Parameters
    ----------
    param1 : int
        Description du param1
    param2 : str
        Description du param2
        
    Returns
    -------
    bool
        Description de la valeur retournée
        
    Raises
    ------
    ValueError
        Si quelque chose ne va pas
    """
    pass
```

### Nommage
- Variables : `snake_case`
- Constantes : `UPPER_SNAKE_CASE`
- Classes : `PascalCase`
- Fonctions privées : `_prefixe_underscore`

## Testing

### Écrire des Tests
Utilisez `pytest` pour les tests unitaires :

```python
# tests/test_indicators.py
import pytest
from src.technical_indicators import calculate_rsi

def test_calculate_rsi():
    """Test le calcul du RSI."""
    # Arrange
    prices = [100, 102, 101, 103, 105]
    
    # Act
    rsi = calculate_rsi(prices, window=2)
    
    # Assert
    assert rsi is not None
    assert len(rsi) == len(prices)
```

### Lancer les Tests
```bash
pytest tests/
pytest --cov=src/  # Avec couverture
```

## Types de Contributions

### 1. Nouvelles Features
- Nouveaux modèles de prévision
- Nouveaux indicateurs techniques
- Nouvelles métri ques de risque
- Fonctionnalités d'optimisation

### 2. Bug Fixes
- Corrections de bugs existants
- Améliorations de performance

### 3. Documentation
- Amélioration du README
- Tutoriels et guides
- Exemples d'utilisation

### 4. Tests
- Tests unitaires
- Tests d'intégration
- Augmentation de la couverture

## Issues et Discussions

### Signaler un Bug
1. Allez dans [Issues](https://github.com/Minyurano/Analyse-quantitative-d-une-action-boursi-re/issues)
2. Cliquez sur "New Issue"
3. Sélectionnez "Bug report"
4. Remplissez le formulaire avec :
   - Description du problème
   - Étapes pour reproduire
   - Comportement attendu
   - Comportement actuel
   - Informations sur l'environnement (Python version, dépendances, etc.)

### Suggérer une Feature
1. Allez dans [Issues](https://github.com/Minyurano/Analyse-quantitative-d-une-action-boursi-re/issues)
2. Cliquez sur "New Issue"
3. Sélectionnez "Feature request"
4. Décrivez votre idée et ses bénéfices

## Processus de Review

1. **Vérification Automatique** : Les tests et la qualité de code sont vérifiés automatiquement
2. **Revue du Mainteneur** : Un revue manuelle est effectuée
3. **Feedback** : Des commentaires ou suggestions peuvent être laissés
4. **Approbation** : La PR est approuvée et fusionnée

## Règles de Conduite

- Soyez respectueux et inclusif
- Acceptez les critiques constructives
- Focalisez-vous sur le code, pas sur la personne
- Aidez les autres contributeurs

## Questions ?

Si vous avez des questions :
- Posez une discussion dans [Discussions](https://github.com/Minyurano/Analyse-quantitative-d-une-action-boursi-re/discussions)
- Ouvrez une issue avec le label `question`
- Contactez les mainteneurs

## Remerciements

Merci de contribuer à ce projet ! Votre contribution est vraiment appréciée.

---

**Bonnes pratiques de commit** :
- Commits atomiques (une seule logique par commit)
- Messages de commit explicites
- Référencez les issues : "Fix #123" pour fermer l'issue automatiquement

**Avant de créer une PR, vérifiez** :
- ✅ Votre code suit les standards
- ✅ Tests unitaires pour les nouvelles features
- ✅ Tous les tests passent
- ✅ Documentation mise à jour
- ✅ Pas de conflits avec la branche main

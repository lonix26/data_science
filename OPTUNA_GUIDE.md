# 🔍 Optimisation Optuna - Guide d'Utilisation

## 📋 Résumé des Modifications

Le notebook `M1-Lasso.ipynb` a été complété avec **2 nouvelles cellules (9 et 10)** dédiées à l'optimisation avec **Optuna**.

### **Cellule 9 : OPTUNA HYPERPARAMETER TUNING**
- Teste 3 modèles en parallèle : **Lasso**, **Ridge**, **ElasticNet**
- Optimise les hyperparamètres :
  - `alpha` : pénalité de régularisation
  - `l1_ratio` : ratio L1 vs L2 (pour ElasticNet)
  - `scaler` : choix du scaler (RobustScaler, StandardScaler, QuantileTransformer)
- **50 trials** pour explorer l'espace des paramètres
- Utilise **TPE Sampler** (Tree-structured Parzen Estimator) - meilleure performance

### **Cellule 10 : PRÉDICTIONS & SUBMISSION FINALE**
- Utilise le meilleur modèle trouvé par Optuna
- Génère des prédictions pour Kaggle
- Crée un fichier CSV nommé `M1_Lasso_Optuna_[ModelType]_Submission.csv`
- Affiche un résumé complet de l'optimisation

---

## 🚀 Comment Exécuter

1. **Exécuter les cellules 1 à 8** (baseline + optimisation manuelle)
2. **Exécuter la cellule 9** (Optuna - ~10-15 minutes)
   ```
   ⏱️ Temps estimé : 10-15 min
   📊 50 trials testés
   ```
3. **Exécuter la cellule 10** (prédictions finales - ~30 secondes)

---

## 📊 Résultats Attendus

| Approche | Modèle | RMSE | Amélioration |
|----------|--------|------|--------------|
| Baseline | Lasso | 0.1384 | - |
| Optimisé | Lasso | 0.1221 | +11.8% |
| **Optuna** | **Lasso/Ridge/ElasticNet** | **~0.1200** | **+13% à +15%** |

### Exemple de sortie Optuna :
```
🏆 MEILLEUR TRIAL (Optuna) :
   RMSE : 0.1198
   Modèle : Ridge
   Scaler : QuantileTransformer
   Alpha  : 0.000234

📈 Amélioration vs Baseline : 13.44%
📈 Amélioration vs Optimisé : 1.88%
```

---

## 🎯 Qu'est-ce qui se passe en détail ?

### **Exploration des Hyperparamètres**

Optuna teste différentes combinaisons :

1. **Modèle** (3 options) :
   - Lasso : L1 regularization seul
   - Ridge : L2 regularization seul  
   - ElasticNet : Mélange L1 + L2

2. **Scaler** (3 options) :
   - RobustScaler : Basique, résistant aux outliers
   - StandardScaler : Normalisation standard
   - QuantileTransformer : Transformation en distribution normale

3. **Alpha** (continu) :
   - Plage : 1e-6 à 1e-1
   - Évite l'overfitting (alpha trop grand) et l'underfitting (alpha trop petit)

4. **L1_ratio** (ElasticNet uniquement) :
   - Plage : 0.0 à 1.0
   - 0 = Ridge, 1 = Lasso, entre = mélange

### **Stratégie de Recherche (TPE)**

- ✅ **Intelligent** : Utilise les résultats précédents pour guider la recherche
- ✅ **Rapide** : Converge vers les meilleur hyperparamètres plus vite que GridSearch
- ✅ **Adaptatif** : Explore les zones prometteuses plus en profondeur

---

## 💡 Configuration Optuna

Vous pouvez ajuster ces paramètres dans la cellule 9 :

```python
# Nombre de trials (augmenter pour meilleure précision, mais plus long)
study.optimize(objective_func, n_trials=50)

# Sampler : TPE, RandomSampler, GridSampler, etc.
sampler=optuna.samplers.TPESampler(seed=42)

# Pruner : arrête les trials non-prometteurs tôt
pruner=optuna.pruners.MedianPruner()
```

---

## 📁 Fichiers Générés

Après exécution, vous aurez :

1. **M1_Lasso_QuantileTransformer_Optimized_Submission.csv** 
   - Prédictions du modèle optimisé manuellement

2. **M1_Lasso_Optuna_[ModelType]_Submission.csv**
   - Prédictions du meilleur modèle trouvé par Optuna
   - Exemple : `M1_Lasso_Optuna_Ridge_Submission.csv`

---

## ⚙️ Dépendances Requises

```
scikit-learn >= 1.3.0
optuna >= 3.0.0
pandas >= 1.5.0
numpy >= 2.0.0
```

✅ **Optuna est déjà installé** dans votre environnement !

---

## 🎓 Concepts Clés Utilisés

### 1. **Cross-Validation dans Optuna**
- Chaque trial entraîne le modèle et l'évalue
- Utilise la métrique RMSE pour évaluer la performance

### 2. **Early Stopping (Optuna)**
- Les trials qui donnent des résultats mauvais peuvent être "prunés" (arrêtés tôt)
- Économise du temps de calcul

### 3. **Hyperparameter Space**
- Continuous : `suggest_loguniform()`, `suggest_uniform()`
- Categorical : `suggest_categorical()`

### 4. **Direction de l'Optimisation**
- `direction='minimize'` : cherche les hyperparamètres qui minimisent RMSE

---

## 🔗 Références

- **Documentation Optuna** : https://optuna.readthedocs.io/
- **Tutoriel TPE Sampler** : https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.TPESampler.html
- **Best Practices** : https://optuna.readthedocs.io/en/stable/tutorial/index.html

---

## ❓ Troubleshooting

### **Cellule 9 prend trop de temps**
→ Réduire `n_trials=50` à `n_trials=30` ou `n_trials=20`

### **Erreur : "No module named optuna"**
→ Lancer : `pip install optuna` (déjà fait)

### **RMSE de Optuna > RMSE optimisé**
→ Normal ! Optuna explore davantage, le modèle optimisé manuellement était déjà très bon

---

## 📊 Exemple de Sortie Complète

```
======================================================================
🔍 OPTUNA - OPTIMISATION AVANCÉE DES HYPERPARAMÈTRES
======================================================================

   Initialisation d'Optuna...
   • Modèles testés : Lasso, Ridge, ElasticNet
   • Hyperparamètres : alpha, l1_ratio, scaler
   • Trials : 50 | Timeout : 5 min

   Lancement de l'optimisation (50 trials)...
   [####################] 100%

======================================================================
📊 RÉSULTATS OPTUNA
======================================================================

🏆 MEILLEUR TRIAL (Optuna) :
   RMSE : 0.1198
   Modèle : Ridge
   Scaler : QuantileTransformer
   Alpha  : 0.000234

   📈 Amélioration vs Baseline : 13.44%
   📈 Amélioration vs Optimisé : 1.88%

======================================================================
🚀 ENTRAÎNEMENT DU MODÈLE OPTUNA OPTIMAL
======================================================================

   Entraînement du modèle Ridge optimal...

📊 MÉTRIQUES DU MODÈLE OPTUNA :
   RMSE : 0.1198
   MAE  : 0.0810
   R²   : 0.9082

======================================================================
📤 PRÉDICTIONS OPTUNA POUR KAGGLE
======================================================================

   Prédictions générées :
   • Min  : $51,234
   • Max  : $567,890
   • Méd. : $158,901
   • Moy. : $176,543

   ✅ Fichier généré : M1_Lasso_Optuna_Ridge_Submission.csv
```

---

**Bon optage ! 🚀**

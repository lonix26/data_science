# Optimisation - Explorations des Leviers

## 📌 Contexte Initial
- **Baseline RMSE** : 0.1384
- **Objectif** : ≤ 0.125
- **Écart à combler** : 0.0134 (0.97%)

---

## 🎯 Stratégie d'Optimisation

Pour améliorer le modèle baseline, trois leviers principaux ont été explorés :

### **Levier 1 : Optimiser le range du paramètre Alpha**

**Hypothèse :** Une résolution plus fine des alphas testés permettrait de trouver un optimum meilleur.

**Configurations testées :**
| Configuration | Résolution | RMSE | Amélioration |
|---|---|---|---|
| Ultra fine (500) | logspace(-5, -1, 500) | 0.1382 | -0.14% |
| Fine (200) | logspace(-4, -2, 200) | 0.1380 | +0.29% |
| Courant (baseline) | Default | 0.1384 | - |

**Résultat :** Amélioration marginale (~0.29%)

---

### **Levier 2 : Augmenter la Validation Croisée**

**Hypothèse :** Augmenter le nombre de folds (CV) réduirait la variance de l'estimation et améliorerait la généralisation.

**Configurations testées :**
| CV Folds | RMSE | Amélioration |
|---|---|---|
| 5 (baseline) | 0.1384 | - |
| 10 | 0.1381 | +0.22% |
| 15 | 0.1380 | +0.29% |

**Résultat :** Amélioration modeste avec CV=15 (~0.29%)

---

### **Levier 3 : Essayer Différents Scalers ⭐ MEILLEUR**

**Hypothèse :** Le choix du scaler impacte fortement la performance, particulièrement pour les données asymétriques comme les prix immobiliers.

**Configurations testées :**
| Scaler | Type | RMSE | Amélioration | Notes |
|---|---|---|---|---|
| **RobustScaler** | Baseline | 0.1384 | - | Résistant aux outliers mais limité |
| StandardScaler | Standard | 0.1382 | +0.14% | Moins performant |
| **QuantileTransformer** | Transformation gaussienne | 0.1221 | **+11.78%** | 🏆 **MEILLEUR** |

**Détail QuantileTransformer :**
- Transforme les données en distribution normale
- Particulièrement efficace pour les données asymétriques
- Combine bien avec Lasso et log-transformation du prix
- Rend le modèle robuste aux outliers

**Résultat :** **Amélioration significative de +11.78%** ✅

---

## 📊 Résumé Comparatif

```
┌─────────────────────────────────────────────────────┐
│         RÉSUMÉ DES LEVIERS D'OPTIMISATION          │
├─────────────────────────────────────────────────────┤
│ Baseline                          0.1384            │
│ ├─ Levier 1 (Alpha)              0.1380 (+0.29%)   │
│ ├─ Levier 2 (CV)                 0.1380 (+0.29%)   │
│ └─ Levier 3 (QuantileTransformer)0.1221 (+11.78%)⭐│
└─────────────────────────────────────────────────────┘

MEILLEUR LEVIER IDENTIFIÉ :
→ QuantileTransformer (output_distribution='normal')
```

---

## 💡 Clés de Succès du Modèle Optimisé

La combinaison suivante s'est avérée optimale :

1. **Log-transformation du target** (`np.log1p(SalePrice)`)
   - Gère l'asymétrie des prix
   - Réduit l'impact des valeurs extrêmes

2. **QuantileTransformer** pour les features numériques
   - Transforme en distribution normale
   - Robustesse aux outliers maintenue
   - Amélioration de +11.78%

3. **TargetEncoder** pour les features catégoriques
   - Encodage par moyenne du target (CV=5)
   - Meilleur que OneHotEncoder pour 43 catégories
   - Capture les relations catégorie-prix

4. **LassoCV** avec fine-tuning d'alpha
   - Regularization naturelle
   - Sélection automatique de features
   - Résistance aux outliers

---

## 🎬 Passage à la Production

Suite aux explorations :
- ✅ Baseline (RobustScaler) : **RMSE = 0.1384**
- ✅ **Optimisé (QuantileTransformer) : RMSE = 0.1221** ← Déployé
- ✅ Optuna (50 trials) : Fine-tuning supplémentaire

**Pipeline final utilisé :** Voir notebook cellule 12 (Modèle optimisé)

---

## 📈 Recommandations Futures

1. **Stacking/Ensemble** : Combiner Lasso + Ridge + ElasticNet
2. **Feature interactions** : Créer des termes d'interaction (ex: TotalArea × ZonePrice)
3. **Géospatial** : Intégrer latitude/longitude comme features
4. **Hyperparameter grid search** : GridSearchCV sur multiple parameters
5. **Cross-validation strategy** : Time-series ou stratified folds si applicable

---

**Date** : Mai 2026  
**Modèle** : Lasso Regression with QuantileTransformer  
**Métrique cible** : RMSE ≤ 0.125 ✅ Atteint

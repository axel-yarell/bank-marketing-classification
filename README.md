# Prédiction de souscription à un dépôt bancaire

Projet réalisé dans le cadre de ma formation en Data Science. L'objectif est de prédire, à partir des données d'une campagne téléphonique (dataset UCI Bank Marketing, environ 11 000 clients), si un client va souscrire à un dépôt à terme.

## Contexte

Les campagnes téléphoniques coûtent cher. Pouvoir identifier en amont les clients les plus susceptibles de souscrire permet de cibler les appels et de réduire les coûts.

## Données

Le dataset contient des informations sur les clients (âge, profession, situation familiale, solde du compte...) et sur les appels (durée, mois, nombre de contacts...). La variable cible est `deposit` (oui/non).

## Démarche

1. Analyse exploratoire : histogrammes des variables numériques, matrice de corrélation
2. Prétraitement : encodage de la variable cible et des variables catégorielles, normalisation avec StandardScaler, split train/test (80/20)
3. Modèle : réseau de neurones (MLP) avec Keras — couches 32 et 16 neurones, Dropout 0.3, sortie sigmoïde
4. Entraînement : 50 époques, batch size 32, validation split 0.2
5. Évaluation : accuracy sur le test, courbes d'apprentissage, test sur un échantillon

## Résultats

Accuracy sur le jeu de test : **0.81** (81 %)

## Ce que j'ai appris

- Mise en place d'un pipeline complet de classification binaire avec Keras
- Gestion des variables catégorielles et normalisation
- Importance de la validation croisée et du suivi des courbes pour détecter le surapprentissage
- L'encodage des variables catégorielles m'a posé problème au début, j'ai fini par tester plusieurs approches avant de garder LabelEncoder.

## Notebook

[Ouvrir dans Colab](https://colab.research.google.com/github/axel-yarell/bank-marketing-classification/blob/main/campagne_de_marketing_bancaire.ipynb)

## Fichiers

- `campagne_de_marketing_bancaire.ipynb` : notebook principal

# 🌿 Sougui Décideurs — Plateforme d'Intelligence Achat & BI

> Application Angular 18 de pilotage décisionnel pour l'artisanat tunisien, sécurisée par Keycloak et enrichie d'analyses prédictives : classification fiscale, estimation TTC et segmentation fournisseurs.

---

## 📋 Table des matières

- [Présentation du projet](#présentation-du-projet)
- [Fonctionnalités](#fonctionnalités)
- [Architecture technique](#architecture-technique)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Configuration Keycloak](#configuration-keycloak)
- [Charte graphique Sougui.tn](#charte-graphique-souguiTn)
- [Structure du projet](#structure-du-projet)
- [Rôles et accès](#rôles-et-accès)
- [Flux d'authentification](#flux-dauthentification)
- [Lancer l'application](#lancer-lapplication)
- [Tests et validation](#tests-et-validation)
- [Déploiement](#déploiement)
- [Équipe](#équipe)

---

## 🎯 Présentation du projet

**Sougui Décideurs** est une plateforme web destinée aux décideurs de l'écosystème artisanal tunisien. Elle centralise six espaces métier :

| Module | Rôle cible |
|--------|------------|
| 🛒 **Achat** | Responsable achats |
| 💼 **Commercial B2C** | Chargé de vente grand public |
| 📊 **Marketing** | Équipe marketing |
| 🏦 **Financier** | Directeur financier |
| 🤝 **Vente B2B** | Responsable partenariats |
| 🎯 **Direction Générale** | General Manager |

Chaque espace propose des outils d'aide à la décision appuyés sur des analyses de données avancées.

---

## ✨ Fonctionnalités

### Module Achat — `/purchase`

**Analyse du régime fiscal**
Détermine si un achat relève du régime TVA standard ou d'un régime réduit/exonéré. Les deux indicateurs clés sont le Taux TVA (52 % d'influence) et la Marge TVA (44 % d'influence).

**Estimation du montant TTC**
Prédit le montant toutes taxes comprises selon le fournisseur, la catégorie d'achat et la période concernée.

**Segmentation fournisseur**
Classe automatiquement un fournisseur dans l'un des trois profils : Petit fournisseur, Fournisseur régulier ou Fournisseur stratégique, selon son historique de volume et de facturation.

### Sécurité transversale

- Authentification SSO centralisée via Keycloak 24
- Contrôle d'accès par rôle métier (RBAC)
- Token JWT automatiquement injecté sur toutes les requêtes HTTP sortantes
- Sidebar dynamique : les liens de navigation sont visibles uniquement selon le profil connecté

---

## 🏛️ Architecture technique

L'application repose sur trois couches principales qui communiquent entre elles.

Le **navigateur Angular 18** héberge les composants métier, la sidebar conditionnelle et les guards de route. Toutes les requêtes sortantes passent par un intercepteur qui attache automatiquement le token JWT.

Le **serveur Keycloak 24.0.5** Keycloak est un "Fournisseur d'Identité et d'Accès" (IAM) qui s'occupe de tout :

Single Sign-On (SSO) : Vos utilisateurs se connectent une seule fois (sur Keycloak) et sont automatiquement reconnus par toutes vos applications (votre app Angular, votre future API, votre outil d'admin...).

Interface d'administration : Vous avez une console graphique pour gérer les utilisateurs, les rôles (financier, achat, etc.), les permissions et les sessions, sans écrire une ligne de code.

Gestion du cycle de vie des utilisateurs : Inscription, email de vérification, réinitialisation de mot de passe, etc. (c'est ce que vous essayez de configurer !).

Supporte des protocoles standards : Keycloak utilise des protocoles comme OpenID Connect (OIDC) , qui est une "couche d'authentification" construite par-dessus OAuth 2.0.

Fédération d'identités : Vous pouvez connecter d'autres sources (comme Google, Facebook, Active Directory) pour que vos utilisateurs puissent s'y authentifier via Keycloak

Le **backend API** reçoit les requêtes sécurisées depuis Angular et expose les endpoints métier ainsi que les modèles d'analyse prédictive.

**Stack principale :**

| Couche | Technologie |
|--------|-------------|
| Framework frontend | Angular 18 — standalone components, signals |
| Authentification IAM | Keycloak 24.0.5 |
| Librairies auth Angular | keycloak-angular 16.6.0 · keycloak-js 24.0.5 |
| Style | SCSS — Playfair Display + DM Sans |
| Gestion d'état | Angular Signals |

---

## 📦 Prérequis

| Outil | Version minimale |
|-------|-----------------|
| Node.js | 20.x LTS |
| npm | 10.x |
| Angular CLI | 18.x |
| Java JDK | 17+ (requis par Keycloak) |
| Keycloak | 24.0.5 |

---

## 🚀 Installation

**Étape 1 — Cloner le dépôt**
Récupérer le projet depuis GitHub sur la branche `feature/keycloak-auth`.
Dépôt : [https://github.com/chahnezelbez/Esprit-PIDEV-4BI5-2026-Ecommerce](https://github.com/chahnezelbez/Esprit-PIDEV-4BI5-2026-Ecommerce)

**Étape 2 — Installer les dépendances**
Exécuter `npm install` à la racine du projet pour installer toutes les dépendances Angular, y compris les packages Keycloak.

**Étape 3 — Lancer le serveur de développement**
Démarrer Angular avec `ng serve`. L'application sera disponible sur `http://localhost:4200` et redirigera automatiquement vers la page de connexion Keycloak si l'utilisateur n'est pas authentifié.

---

## 🔐 Configuration Keycloak

### Démarrer le serveur Keycloak

Keycloak se lance en mode développement via le script `kc.bat start-dev` (Windows) ou `kc.sh start-dev` (Linux/macOS). Si le port 8080 est occupé, ajouter l'option `--http-port=8180`. La console d'administration est alors accessible sur `http://localhost:8080`.

### Créer le Realm

Créer un nouveau Realm nommé `sougui-realm` depuis la console d'administration Keycloak.

### Créer le Client Angular

Créer un client nommé `angular-app` avec la configuration suivante :

| Paramètre | Valeur |
|-----------|--------|
| Type de client | Public |
| Protocole | OpenID Connect |
| Valid redirect URIs | `http://localhost:4200/*` |
| Valid post logout redirect URIs | `http://localhost:4200/*` |
| Web origins | `http://localhost:4200` |
| Flux d'authentification | Standard flow uniquement |

### Rôles métier à créer dans Keycloak

| Rôle | Route associée |
|------|----------------|
| `achat` | `/purchase` |
| `vente_b2c` | `/commercial` |
| `marketing` | `/marketing` |
| `general_manager` | `/gm` |
| `vente_b2b` | `/b2b` |
| `financier` | `/financier` |

Chaque utilisateur est créé dans le realm et se voit attribuer un ou plusieurs rôles correspondant à son périmètre métier.

---

## 🎨 Charte graphique Sougui.tn

La palette reflète l'identité artisanale tunisienne fusionnée avec une esthétique BI/tech moderne.

| Rôle | Couleur | Code hexadécimal |
|------|---------|-----------------|
| Couleur primaire — artisanat & confiance | 🟢 Vert méditerranéen | `#1F6F5B` |
| Accent chaud — tradition tunisienne | 🟠 Terre cuite | `#E07A3F` |
| Fond principal — style artisanal premium | 🟡 Beige sable | `#F5E9DA` |
| Tech / BI / Data | 🔵 Bleu profond | `#1E3A8A` |
| Boutons CTA | ✨ Doré premium | `#D4A017` |
| Texte principal | ⚫ Noir soft | `#111827` |
| Texte secondaire | 🩶 Gris | `#6B7280` |

**Typographie :**
- Titres et en-têtes : **Playfair Display** — serif, élégant, identitaire
- Corps de texte et UI : **DM Sans** — moderne, lisible, neutre

**Principes UX :**
- Fond beige sable pour les pages e-commerce et formulaires
- Vert méditerranéen pour la navigation principale et les éléments de validation
- Doré pour tous les boutons d'action principaux
- Bleu profond réservé aux tableaux de bord et graphiques analytiques

---

## 🗂️ Structure du projet

Le projet est organisé selon les conventions Angular standalone avec une séparation claire entre les préoccupations d'authentification et les composants métier.

| Dossier / Fichier | Rôle |
|-------------------|------|
| `app/keycloak/` | Initialisation du client Keycloak |
| `app/services/` | Service de lecture des rôles JWT |
| `app/guards/` | Guard RBAC paramétrable par rôle |
| `app/components/purchase/` | Module Achat — analyse, estimation, segmentation |
| `app/components/commercial/` | Module Commercial B2C |
| `app/components/marketing/` | Module Marketing |
| `app/components/financier/` | Module Financier |
| `app/components/b2b/` | Module Vente B2B |
| `app/components/gm/` | Module Direction Générale |
| `app/app.component` | Sidebar conditionnelle + bouton déconnexion |
| `app/app.config` | Configuration application + initialisation Keycloak |
| `app/app.routes` | Routes protégées par guards de rôle |
| `styles/theme.scss` | Variables SCSS de la charte Sougui.tn |
| `assets/silent-check-sso.html` | Page pour l'authentification SSO silencieuse |

---

## 👥 Rôles et accès

### Tableau des permissions

| Rôle Keycloak | Route | Fonctionnalités accessibles |
|---------------|-------|-----------------------------|
| `achat` | `/purchase` | Analyse fiscale, Estimation TTC, Segmentation fournisseurs |
| `vente_b2c` | `/commercial` | Tableau de bord ventes, indicateurs B2C |
| `marketing` | `/marketing` | Campagnes, analytics audience |
| `financier` | `/financier` | Bilans, trésorerie, ratios financiers |
| `vente_b2b` | `/b2b` | Gestion partenariats, contrats |
| `general_manager` | `/gm` | Vue consolidée — accès à tous les modules |

> Un utilisateur sans le bon rôle qui tente d'accéder directement à une route est automatiquement redirigé vers `/purchase`.

### Fonctionnement du guard

Le guard `roleGuard` est paramétrable : il reçoit la liste des rôles autorisés pour chaque route. Il vérifie d'abord que l'utilisateur est authentifié, puis contrôle qu'il possède au moins un des rôles requis. En cas d'échec, il déclenche respectivement une redirection vers le login Keycloak ou vers une page non autorisée.

### Sidebar conditionnelle

Le composant principal affiche chaque lien de navigation uniquement si le service `KeycloakRoleService` confirme que l'utilisateur courant possède le rôle correspondant. Le bouton « Déconnexion » termine la session Keycloak et redirige vers l'écran de connexion.

---

## 🔁 Flux d'authentification

**Étape 1 — Accès à l'application**
L'utilisateur accède à `http://localhost:4200`. L'initialiseur d'application tente de charger Keycloak avant l'affichage de l'interface.

**Étape 2 — Redirection vers le login**
Si l'utilisateur n'est pas encore authentifié, Keycloak le redirige automatiquement vers la page de connexion du realm `sougui-realm`.

**Étape 3 — Authentification**
L'utilisateur saisit ses identifiants. Keycloak valide le compte et émet un token JWT contenant les rôles attribués.

**Étape 4 — Retour dans l'application**
Keycloak redirige l'utilisateur vers l'URL de retour. Le guard de route lit les rôles depuis le token et autorise ou refuse l'accès à la page demandée.

**Étape 5 — Navigation sécurisée**
Le token JWT est automatiquement attaché à chaque requête HTTP sortante via l'intercepteur intégré. Aucune configuration manuelle n'est nécessaire côté composants.

**Étape 6 — Déconnexion**
Le clic sur « Déconnexion » termine la session Keycloak côté serveur et côté navigateur, puis redirige vers l'écran de login.

---

## ▶️ Lancer l'application

Deux processus doivent tourner simultanément dans deux terminaux distincts.

**Terminal 1 — Keycloak**
Depuis le dossier d'installation de Keycloak, lancer le serveur en mode développement via `kc.bat start-dev` (Windows) ou `kc.sh start-dev` (Linux/macOS).

**Terminal 2 — Angular**
Depuis la racine du projet Angular, exécuter `ng serve --open`. Le navigateur s'ouvre automatiquement sur `http://localhost:4200`.

---

## 🧪 Tests et validation

### Utilisateurs de test recommandés

| Utilisateur | Rôle | Accès attendu |
|-------------|------|---------------|
| `user.achat` | `achat` | `/purchase` uniquement |
| `user.b2b` | `vente_b2b` | `/b2b` uniquement |
| `user.finance` | `financier` | `/financier` uniquement |
| `user.gm` | `general_manager` | Tous les modules |

### Scénarios de validation

- ✅ Connexion avec `user.achat` → seul le lien « Achat » est visible dans la sidebar
- ✅ Tentative d'accès direct à `/b2b` avec `user.achat` → redirection automatique
- ✅ Connexion avec `user.gm` → tous les liens sont visibles et accessibles
- ✅ Clic sur « Déconnexion » → session terminée, retour au login Keycloak
- ✅ Token JWT présent dans les en-têtes des requêtes API (vérifiable via les DevTools du navigateur, onglet Réseau)

---

## 🌐 Déploiement

### Points d'attention pour la production

| Élément | Action requise |
|---------|----------------|
| URL Keycloak | Remplacer `localhost:8180` par l'URL de production `https://auth.sougui.tn` |
| Mode d'initialisation | Passer de `login-required` à `check-sso` pour l'authentification silencieuse |
| HTTPS | Activer les certificats TLS sur Keycloak et l'API backend |
| Redirect URIs | Mettre à jour dans la console Keycloak avec les domaines de production |
| Durée des tokens | Configurer l'expiration des access tokens et refresh tokens selon la politique sécurité |
| CORS | Vérifier et mettre à jour la configuration Web Origins dans Keycloak |
| Variables d'environnement | Utiliser le fichier `environment.prod.ts` pour séparer les configurations dev et prod |

### Build de production

Générer les artefacts avec `ng build --configuration production`. Les fichiers compilés et optimisés seront placés dans le dossier `/dist/sougui-decideurs/` et peuvent être déployés sur n'importe quel serveur web statique (Nginx, Apache, Vercel, etc.).

---

## 👨‍💻 Équipe

| Rôle | Responsable |
|------|-------------|
| Développement Frontend | Équipe PIDEV 4BI5 |
| Administration Keycloak & DevOps | Responsable sécurité |
| Design & Identité visuelle | Sougui.tn Design System |

**Dépôt GitHub :** [https://github.com/chahnezelbez/Esprit-PIDEV-4BI5-2026-Ecommerce](https://github.com/chahnezelbez/Esprit-PIDEV-4BI5-2026-Ecommerce)

**Branche Keycloak :** `feature/keycloak-auth`

---

## 📄 Licence

Projet académique — **ESPRIT School of Engineering**, filière 4BI5 · Année universitaire 2025–2026.

---

*Dernière mise à jour : Mai 2026 · Branche `feature/keycloak-auth`*
"""
SwissShop — Dataset synthétique + Analyse Apriori complète
Génère le dataset, applique les règles d'association, produit les visualisations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ─────────────────────────────────────────────────────────────
# 1. GÉNÉRATION DU DATASET SWISSSHOP
# ─────────────────────────────────────────────────────────────

# Catalogue produits SwissShop (réaliste pour une boutique suisse tech/lifestyle)
CATALOGUE = {
    "Informatique": [
        "Laptop Pro 15\"", "Laptop Ultrabook", "Souris sans fil", "Clavier mécanique",
        "Clavier sans fil", "Webcam HD", "Hub USB-C", "Disque SSD externe",
        "Écran 27\" 4K", "Casque Bluetooth"
    ],
    "Téléphonie": [
        "Smartphone Android", "Smartphone iPhone", "Coque protection",
        "Chargeur rapide", "Câble USB-C", "Protection écran", "Batterie externe"
    ],
    "Audio": [
        "Écouteurs True Wireless", "Enceinte Bluetooth", "Casque HiFi",
        "Adaptateur Jack", "Câble Audio optique"
    ],
    "Bureau": [
        "Imprimante laser", "Cartouche toner", "Papier A4 500f",
        "Lampe de bureau LED", "Support laptop", "Tapis de souris XL"
    ],
    "Accessoires": [
        "Sac à dos PC", "Housse laptop 15\"", "Multiprise USB",
        "Nettoyant écran", "Câble HDMI 2.1", "Lecteur carte SD"
    ]
}

# Règles d'achat réalistes (basket patterns)
# Format: (produits_déclencheurs, produits_associés, probabilité_bonus)
PATTERNS = [
    # Workstation setup
    (["Laptop Pro 15\""],         ["Souris sans fil", "Hub USB-C"],       0.70),
    (["Laptop Pro 15\""],         ["Clavier mécanique"],                  0.45),
    (["Laptop Pro 15\""],         ["Sac à dos PC"],                       0.55),
    (["Laptop Ultrabook"],        ["Housse laptop 15\"", "Souris sans fil"], 0.65),
    (["Laptop Ultrabook"],        ["Hub USB-C"],                          0.60),
    # Écran + périphériques
    (["Écran 27\" 4K"],           ["Câble HDMI 2.1"],                     0.85),
    (["Écran 27\" 4K"],           ["Clavier sans fil", "Souris sans fil"], 0.50),
    # Téléphonie
    (["Smartphone Android"],      ["Coque protection", "Protection écran"], 0.75),
    (["Smartphone Android"],      ["Chargeur rapide"],                    0.60),
    (["Smartphone iPhone"],       ["Coque protection"],                   0.80),
    (["Smartphone iPhone"],       ["Protection écran", "Câble USB-C"],    0.65),
    (["Chargeur rapide"],         ["Câble USB-C"],                        0.78),
    # Audio
    (["Casque Bluetooth"],        ["Adaptateur Jack"],                    0.40),
    (["Enceinte Bluetooth"],      ["Câble Audio optique"],                0.35),
    # Bureau
    (["Imprimante laser"],        ["Cartouche toner", "Papier A4 500f"], 0.90),
    (["Imprimante laser"],        ["Câble USB-C"],                        0.55),
    # Combos fréquents
    (["Souris sans fil", "Clavier sans fil"], ["Tapis de souris XL"],     0.55),
    (["Laptop Pro 15\"", "Écran 27\" 4K"],    ["Support laptop"],         0.60),
    (["Disque SSD externe"],      ["Câble USB-C"],                        0.70),
    (["Webcam HD"],               ["Casque Bluetooth"],                   0.45),
    (["Batterie externe"],        ["Câble USB-C"],                        0.80),
]

ALL_PRODUCTS = [p for cat in CATALOGUE.values() for p in cat]

def generate_transaction():
    """Génère une transaction avec patterns réalistes."""
    basket = set()
    
    # Produit initial aléatoire
    start = np.random.choice(ALL_PRODUCTS,
                              p=[1/len(ALL_PRODUCTS)] * len(ALL_PRODUCTS))
    basket.add(start)
    
    # Appliquer les patterns associés
    for triggers, associated, prob in PATTERNS:
        if any(t in basket for t in triggers):
            for product in associated:
                if np.random.random() < prob:
                    basket.add(product)
    
    # Ajout de produits aléatoires supplémentaires (bruit réaliste)
    n_extra = np.random.choice([0, 1, 2], p=[0.60, 0.30, 0.10])
    for _ in range(n_extra):
        basket.add(np.random.choice(ALL_PRODUCTS))
    
    return list(basket)

# Générer 3000 transactions
N_TRANSACTIONS = 3000
print(f"⏳ Génération de {N_TRANSACTIONS} transactions SwissShop...")

transactions_list = [generate_transaction() for _ in range(N_TRANSACTIONS)]

# Construire le DataFrame raw
records = []
for tid, items in enumerate(transactions_list, 1):
    date = pd.Timestamp('2024-01-01') + pd.Timedelta(
        days=np.random.randint(0, 365))
    customer = f"C{np.random.randint(1000, 9999)}"
    for item in items:
        # Trouver la catégorie
        cat = next((c for c, prods in CATALOGUE.items() if item in prods), "Autre")
        records.append({
            "transaction_id": f"T{tid:05d}",
            "customer_id": customer,
            "date": date,
            "product_name": item,
            "category": cat,
            "price_chf": round(np.random.uniform(9.90, 1299.00), 2)
        })

df_raw = pd.DataFrame(records)

print(f"✅ Dataset généré : {len(df_raw):,} lignes | {df_raw['transaction_id'].nunique():,} transactions | {df_raw['product_name'].nunique()} produits distincts")
print(f"   Période : {df_raw['date'].min().date()} → {df_raw['date'].max().date()}")
print(f"\n{df_raw.head(8).to_string(index=False)}\n")

# Sauvegarder le dataset
df_raw.to_csv('/home/claude/swissshop_transactions.csv', index=False)
print("💾 Dataset sauvegardé : swissshop_transactions.csv\n")

# ─────────────────────────────────────────────────────────────
# 2. TRANSFORMATION EN FORMAT BASKET
# ─────────────────────────────────────────────────────────────

print("⏳ Transformation en format basket...")
basket = (
    df_raw
    .groupby(['transaction_id', 'product_name'])['product_name']
    .count()
    .unstack(fill_value=0)
    .astype(bool)
)
print(f"✅ Basket matrix : {basket.shape[0]} transactions × {basket.shape[1]} produits\n")

# ─────────────────────────────────────────────────────────────
# 3. APRIORI + RÈGLES D'ASSOCIATION
# ─────────────────────────────────────────────────────────────

print("⏳ Calcul des itemsets fréquents (Apriori, min_support=0.01)...")
frequent_itemsets = apriori(
    basket,
    min_support=0.01,
    use_colnames=True,
    max_len=4
)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(len)
print(f"✅ {len(frequent_itemsets)} itemsets fréquents trouvés\n")

print("⏳ Génération des règles d'association...")
rules = association_rules(
    frequent_itemsets,
    metric="lift",
    min_threshold=1.2,
)


rules_filtered = rules[
    (rules['confidence'] >= 0.40) &
    (rules['lift']       >= 1.5)  &
    (rules['support']    >= 0.01)
].sort_values('lift', ascending=False).reset_index(drop=True)

print(f"✅ {len(rules_filtered)} règles extraites après filtrage\n")

# Aperçu des top règles
rules_display = rules_filtered.copy()
rules_display['antecedents'] = rules_display['antecedents'].apply(lambda x: ', '.join(list(x)))
rules_display['consequents'] = rules_display['consequents'].apply(lambda x: ', '.join(list(x)))
print("TOP 10 RÈGLES (triées par lift) :")
print(rules_display[['antecedents','consequents','support','confidence','lift']].head(10).to_string(index=False))
print()

# ─────────────────────────────────────────────────────────────
# 4. VISUALISATION 1 — HEATMAP SUPPORT × CONFIANCE × LIFT
# ─────────────────────────────────────────────────────────────

print("⏳ Génération de la heatmap...")

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.patch.set_facecolor('#0f1923')

top_rules = rules_filtered.head(25).copy()
top_rules['antecedents_str'] = top_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
top_rules['consequents_str'] = top_rules['consequents'].apply(lambda x: ', '.join(list(x)))
top_rules['rule_label'] = top_rules['antecedents_str'] + '\n→ ' + top_rules['consequents_str']

# ── Panneau gauche : Scatter Support × Confiance (couleur = lift, taille = lift)
ax1 = axes[0]
ax1.set_facecolor('#1a2535')

scatter = ax1.scatter(
    top_rules['support'],
    top_rules['confidence'],
    c=top_rules['lift'],
    s=top_rules['lift'] ** 2.2 * 15,
    cmap='RdYlGn',
    alpha=0.85,
    edgecolors='white',
    linewidths=0.5,
    vmin=1.0, vmax=top_rules['lift'].max()
)

# Colorbar
cbar = plt.colorbar(scatter, ax=ax1)
cbar.set_label('Lift', color='white', fontsize=11)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

# Zones de qualité
ax1.axhline(y=0.60, color='#f0c040', linestyle='--', linewidth=1, alpha=0.6, label='Confiance 60%')
ax1.axvline(x=0.02, color='#40c0f0', linestyle='--', linewidth=1, alpha=0.6, label='Support 2%')

# Annotations des 6 meilleures règles
for i, row in top_rules.head(6).iterrows():
    ax1.annotate(
        row['rule_label'],
        (row['support'], row['confidence']),
        fontsize=6.5, color='white', alpha=0.9,
        xytext=(8, 5), textcoords='offset points',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#2a3a50', alpha=0.7)
    )

ax1.set_xlabel('Support', color='white', fontsize=12)
ax1.set_ylabel('Confiance', color='white', fontsize=12)
ax1.set_title('Vue d\'ensemble des règles\nSupport × Confiance (taille/couleur = Lift)',
              color='white', fontsize=12, pad=15)
ax1.tick_params(colors='white')
ax1.spines[:].set_color('#3a4a5a')
ax1.legend(fontsize=8, facecolor='#1a2535', labelcolor='white', framealpha=0.7)

# ── Panneau droit : Bar chart horizontal des top règles par lift
ax2 = axes[1]
ax2.set_facecolor('#1a2535')

top15 = top_rules.head(15).sort_values('lift')
colors_lift = plt.cm.RdYlGn(
    (top15['lift'] - top15['lift'].min()) /
    (top15['lift'].max() - top15['lift'].min())
)

bars = ax2.barh(
    range(len(top15)),
    top15['lift'],
    color=colors_lift,
    edgecolor='white',
    linewidth=0.5,
    height=0.7
)

# Ligne de référence lift = 1
ax2.axvline(x=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label='Lift = 1 (hasard)')

# Labels sur les barres
for i, (bar, row) in enumerate(zip(bars, top15.itertuples())):
    ax2.text(
        bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
        f'{row.lift:.2f}× | conf: {row.confidence:.0%}',
        va='center', color='white', fontsize=7.5
    )

# Labels des règles
ax2.set_yticks(range(len(top15)))
ax2.set_yticklabels(
    [f"{r.antecedents_str}\n→ {r.consequents_str}"
     for r in top15.itertuples()],
    fontsize=7, color='white'
)

ax2.set_xlabel('Lift', color='white', fontsize=12)
ax2.set_title('Top 15 règles par Lift\n(corrélation vs. hasard)',
              color='white', fontsize=12, pad=15)
ax2.tick_params(colors='white')
ax2.spines[:].set_color('#3a4a5a')
ax2.legend(fontsize=8, facecolor='#1a2535', labelcolor='white', framealpha=0.7)
ax2.set_xlim(0, top15['lift'].max() * 1.25)

fig.suptitle(
    'SwissShop — Analyse des Règles d\'Association (Algorithme Apriori)\n'
    f'{len(rules_filtered)} règles extraites | {N_TRANSACTIONS:,} transactions | min_support=1% | min_confidence=40% | min_lift=1.5',
    color='white', fontsize=13, y=1.01
)

plt.tight_layout(pad=2.0)
plt.savefig('/home/claude/swissshop_heatmap.png', dpi=150,
            bbox_inches='tight', facecolor='#0f1923')
plt.close()
print("✅ Heatmap sauvegardée : swissshop_heatmap.png\n")

# ─────────────────────────────────────────────────────────────
# 5. VISUALISATION 2 — RÉSEAU D'ASSOCIATIONS
# ─────────────────────────────────────────────────────────────

print("⏳ Génération du graphe réseau...")

G = nx.DiGraph()
top_net = rules_filtered.head(20).copy()

for _, row in top_net.iterrows():
    ante = ', '.join(list(row['antecedents']))
    cons = ', '.join(list(row['consequents']))
    G.add_edge(ante, cons,
               weight=row['lift'],
               confidence=row['confidence'],
               support=row['support'])

fig, ax = plt.subplots(figsize=(16, 11))
fig.patch.set_facecolor('#0f1923')
ax.set_facecolor('#0f1923')

pos = nx.spring_layout(G, k=3.5, seed=42, iterations=50)

# Degré des nœuds pour la taille
degrees = dict(G.degree())
node_sizes = [max(800, degrees[n] * 600) for n in G.nodes()]

# Couleur des nœuds selon leur catégorie
node_colors = []
for node in G.nodes():
    found_cat = "Autre"
    for cat, prods in CATALOGUE.items():
        if any(p in node for p in prods):
            found_cat = cat
            break
    color_map = {
        "Informatique": "#4a9eff",
        "Téléphonie":   "#ff6b6b",
        "Audio":        "#ffd93d",
        "Bureau":       "#6bcb77",
        "Accessoires":  "#c77dff",
        "Autre":        "#aaaaaa"
    }
    node_colors.append(color_map.get(found_cat, "#aaaaaa"))

edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
edge_normalized = [(w - min(edge_weights)) / (max(edge_weights) - min(edge_weights) + 0.001)
                   for w in edge_weights]

nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                       node_color=node_colors, alpha=0.9, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=7,
                        font_color='white', font_weight='bold', ax=ax)
nx.draw_networkx_edges(G, pos,
                       width=[1 + w * 3 for w in edge_normalized],
                       edge_color=[plt.cm.YlOrRd(w) for w in edge_normalized],
                       arrows=True, arrowsize=15,
                       connectionstyle='arc3,rad=0.1', ax=ax)

# Légende catégories
legend_elements = [
    mpatches.Patch(color="#4a9eff", label="Informatique"),
    mpatches.Patch(color="#ff6b6b", label="Téléphonie"),
    mpatches.Patch(color="#ffd93d", label="Audio"),
    mpatches.Patch(color="#6bcb77", label="Bureau"),
    mpatches.Patch(color="#c77dff", label="Accessoires"),
]
ax.legend(handles=legend_elements, loc='lower left',
          facecolor='#1a2535', labelcolor='white', fontsize=9, framealpha=0.8)

ax.set_title(
    'SwissShop — Réseau des Règles d\'Association\n'
    'Épaisseur/couleur des arêtes = Lift | Taille des nœuds = Connectivité',
    color='white', fontsize=13, pad=15
)
ax.axis('off')
plt.tight_layout()
plt.savefig('/home/claude/swissshop_network.png', dpi=150,
            bbox_inches='tight', facecolor='#0f1923')
plt.close()
print("✅ Réseau sauvegardé : swissshop_network.png\n")

# ─────────────────────────────────────────────────────────────
# 6. RÈGLES BUSINESS EN LANGAGE NATUREL
# ─────────────────────────────────────────────────────────────

print("=" * 70)
print("  RÈGLES BUSINESS EXTRAITES — SWISSSHOP")
print("=" * 70)

for i, row in rules_filtered.head(12).iterrows():
    ante = ', '.join(list(row['antecedents']))
    cons = ', '.join(list(row['consequents']))
    print(f"\n📌 Règle #{i+1}")
    print(f"   SI  : {ante}")
    print(f"   ALORS: {cons}")
    print(f"   ├─ Confiance : {row['confidence']:.1%}")
    print(f"   ├─ Support   : {row['support']:.2%}")
    print(f"   └─ Lift      : {row['lift']:.2f}×")
    if row['lift'] > 3.0 and row['confidence'] > 0.55:
        print(f"   💡 Cross-sell PRIORITAIRE ✅")
    elif row['lift'] > 2.0:
        print(f"   💡 Bundle promotionnel recommandé ⚡")
    else:
        print(f"   💡 Placement conjoint en rayon")

print("\n✅ Analyse complète terminée.")
print("   Fichiers générés :")
print("   • swissshop_transactions.csv")
print("   • swissshop_heatmap.png")
print("   • swissshop_network.png")

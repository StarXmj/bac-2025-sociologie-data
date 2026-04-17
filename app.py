import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import unicodedata
import re

# ==========================================
# CONFIGURATION DE LA PAGE
# ==========================================
st.set_page_config(page_title="Data Bac 2025", page_icon="🎓", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    [data-testid="stMetric"] { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; }
    [data-testid="stMetricLabel"] * { color: #555555 !important; font-weight: bold; }
    [data-testid="stMetricValue"] * { color: #000000 !important; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# CHARGEMENT ET CALCULS (Mis en cache)
# ==========================================
@st.cache_data
def load_and_process_all():
    df = pd.read_csv("resultats_bac_2025_complet_geo.csv")

    def get_fname(n):
        m = str(n).split()
        return m[1].capitalize() if len(m) > 1 else m[0].capitalize() if len(m) == 1 else "Inconnu"

    df['Prénom usuel'] = df['Nom - Prénom'].apply(get_fname)

    def get_note(m):
        m = str(m).lower()
        if 'félicitations' in m: return 5
        if 'très bien' in m: return 4
        if 'bien' in m: return 3
        if 'assez bien' in m: return 2
        return 1

    df['Note'] = df['Mention'].fillna('admis').apply(get_note)
    df['Excellence'] = df['Mention'].fillna('admis').apply(lambda x: 1 if 'très' in str(x).lower() or 'fél' in str(x).lower() else 0)

    # --- 1. STATS PAR PRÉNOM ---
    stats = df.groupby('Prénom usuel').agg(
        Note_Moyenne=('Note', 'mean'),
        Proportion_Excellence=('Excellence', 'mean'),
        Total_Candidats=('Excellence', 'count')
    ).reset_index()

    stats = stats[stats['Total_Candidats'] >= 10].copy()
    stats['Taux_TB_Pct'] = stats['Proportion_Excellence'] * 100

    scaler = StandardScaler()
    X = scaler.fit_transform(stats[['Note_Moyenne', 'Taux_TB_Pct']])
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10).fit(X)
    stats['Cluster_Brut'] = kmeans.predict(X)

    order = stats.groupby('Cluster_Brut')['Note_Moyenne'].mean().sort_values().index
    mapping = {old: new for new, old in enumerate(order)}
    stats['Cluster_ID'] = stats['Cluster_Brut'].map(mapping)

    noms_pro = ["Performances Faibles", "Moyenne Basse", "Niveau Médian", "Performances Élevées", "Excellence Académique"]
    couleurs = ['#FF3333', '#FF9100', '#FFD700', '#00C853', '#2962FF']
    
    stats['Famille'] = stats['Cluster_ID'].apply(lambda x: noms_pro[x])
    stats['Couleur'] = stats['Cluster_ID'].apply(lambda x: couleurs[x])

    def phonetique(p):
        p = unicodedata.normalize('NFD', p.lower()).encode('ascii', 'ignore').decode("utf-8")
        p = p.replace('ph', 'f').replace('th', 't').replace('ch', 'c').replace('y', 'i')
        p = re.sub(r'[^a-z]', '', p)
        p = re.sub(r'(.)\1+', r'\1', p)
        return p

    stats['Base'] = stats['Prénom usuel'].apply(phonetique)
    
    def estimer_genre(prenom):
        p = str(prenom).lower().strip()
        score = 0.0
        if p.endswith(('a', 'e', 'ine', 'ette', 'elle', 'ie', 'ia', 'isa', 'ly')): score = 0.8
        elif p.endswith(('o', 'i', 'is', 'us', 'ic', 'al', 'el', 'ien', 'on', 'an', 'en', 'er', 'ur')): score = -0.8
        
        valeur_ascii = sum([ord(c) for c in p])
        bruit = (valeur_ascii % 50) / 100.0  
        
        if score > 0: score = min(1.0, score - 0.2 + bruit)
        elif score < 0: score = max(-1.0, score + 0.2 - bruit)
        else: score = -0.5 + bruit 
        return score

    stats['Score_Genre'] = stats['Prénom usuel'].apply(estimer_genre)

    # --- 2. STATS PAR ACADÉMIE ---
    stats_acad = df.groupby('Académie').agg(
        Note_Moyenne=('Note', 'mean'),
        Proportion_Excellence=('Excellence', 'mean'),
        Total_Candidats=('Excellence', 'count')
    ).reset_index()
    
    stats_acad = stats_acad[stats_acad['Total_Candidats'] >= 10].copy()
    stats_acad['Taux_TB_Pct'] = stats_acad['Proportion_Excellence'] * 100
    
    X_acad = scaler.transform(stats_acad[['Note_Moyenne', 'Taux_TB_Pct']])
    stats_acad['Cluster_ID'] = pd.Series(kmeans.predict(X_acad)).map(mapping).values
    stats_acad['Famille'] = stats_acad['Cluster_ID'].apply(lambda x: noms_pro[x])
    stats_acad['Couleur'] = stats_acad['Cluster_ID'].apply(lambda x: couleurs[x])

    # --- 3. STATS GÉOGRAPHIQUES (Points Cardinaux) ---
    df_geo = df.dropna(subset=['Latitude', 'Longitude']).copy()
    df_geo['Latitude'] = pd.to_numeric(df_geo['Latitude'], errors='coerce')
    df_geo['Longitude'] = pd.to_numeric(df_geo['Longitude'], errors='coerce')
    df_geo.dropna(subset=['Latitude', 'Longitude'], inplace=True)
    
    lat_med = df_geo['Latitude'].median()
    lon_med = df_geo['Longitude'].median()
    
    dy = df_geo['Latitude'] - lat_med
    dx = df_geo['Longitude'] - lon_med
    angles = np.arctan2(dy, dx)
    
    conditions = [
        (angles >= -np.pi/4) & (angles < np.pi/4),
        (angles >= np.pi/4) & (angles < 3*np.pi/4),
        (angles >= 3*np.pi/4) | (angles < -3*np.pi/4),
        (angles >= -3*np.pi/4) & (angles < -np.pi/4)
    ]
    choices = ['EST', 'NORD', 'OUEST', 'SUD']
    df_geo['Cardinal'] = np.select(conditions, choices, default='Inconnu')
    
    stats_cardinal = df_geo.groupby('Cardinal').agg(
        Note_Moyenne=('Note', 'mean'),
        Proportion_Excellence=('Excellence', 'mean'),
        Total_Candidats=('Excellence', 'count')
    ).reset_index()
    
    stats_cardinal = stats_cardinal[stats_cardinal['Cardinal'] != 'Inconnu'].copy()
    stats_cardinal['Taux_TB_Pct'] = stats_cardinal['Proportion_Excellence'] * 100
    
    X_card = scaler.transform(stats_cardinal[['Note_Moyenne', 'Taux_TB_Pct']])
    stats_cardinal['Cluster_ID'] = pd.Series(kmeans.predict(X_card)).map(mapping).values
    stats_cardinal['Famille'] = stats_cardinal['Cluster_ID'].apply(lambda x: noms_pro[x])
    stats_cardinal['Couleur'] = stats_cardinal['Cluster_ID'].apply(lambda x: couleurs[x])

    return stats, kmeans, scaler, noms_pro, couleurs, stats_acad, stats_cardinal

stats_all, model, sc, labels_pro, colors_pro, stats_acad, stats_cardinal = load_and_process_all()

# --- CALCUL DES MOYENNES NATIONALES ---
moy_nat_x = stats_all['Note_Moyenne'].mean()
moy_nat_y = stats_all['Taux_TB_Pct'].mean()

stats_all['Texte_Survol'] = (
    "<b>Prénom : " + stats_all['Prénom usuel'] + "</b><br>" +
    "👥 Candidats : " + stats_all['Total_Candidats'].astype(str) + "<br>" +
    "🎓 Note Moy. : " + stats_all['Note_Moyenne'].round(2).astype(str) + " / 5<br>" +
    "⭐ Taux TB : " + stats_all['Taux_TB_Pct'].round(1).astype(str) + " %<br>" +
    "📍 Groupe : " + stats_all['Famille'] + "<br>" +
    "🚻 Probabilité Fille : " + ((stats_all['Score_Genre'] + 1) * 50).round(0).astype(str) + " %"
)

# ==========================================
# INTERFACE STREAMLIT
# ==========================================
st.title("📊 Bac 2025 : Sociologie, Géographie & Data Science")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Prénom", "📂 Orthographe", "🚻 Genre", "📍 Académies & Géo", "📖 Méthodologie"])

x_min, x_max = stats_all['Note_Moyenne'].min() - 0.2, stats_all['Note_Moyenne'].max() + 0.2
y_min, y_max = stats_all['Taux_TB_Pct'].min() - 5, stats_all['Taux_TB_Pct'].max() + 5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = model.predict(sc.transform(np.c_[xx.ravel(), yy.ravel()]))
notes_c = stats_all.groupby('Cluster_Brut')['Note_Moyenne'].mean().sort_values().index
map_z = {old: new for new, old in enumerate(notes_c)}
Z_mapped = np.vectorize(map_z.get)(Z).reshape(xx.shape)
c_scale = [[i/4, colors_pro[i]] for i in range(5)]

def ajouter_lignes_moyennes(fig):
    fig.add_vline(x=moy_nat_x, line_dash="dot", line_color="rgba(0,0,0,0.4)", line_width=2, 
                  annotation_text="Moy. Nat. Note", annotation_position="top left", annotation_font=dict(color="black", size=11))
    fig.add_hline(y=moy_nat_y, line_dash="dot", line_color="rgba(0,0,0,0.4)", line_width=2, 
                  annotation_text="Moy. Nat. TB", annotation_position="bottom right", annotation_font=dict(color="black", size=11))
    return fig

# ==========================================
# ONGLET 1 : RECHERCHE INDIVIDUELLE
# ==========================================
with tab1:
    q = st.text_input("Tapez votre prénom :", placeholder="Ex: Léo, Camille...").strip().capitalize()
    if q:
        row = stats_all[stats_all['Prénom usuel'] == q]
        if row.empty:
            st.warning("Prénom non trouvé ou moins de 10 candidats.")
        else:
            d = row.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Candidats", int(d['Total_Candidats']))
            c2.metric("Note Moyenne", f"{d['Note_Moyenne']:.2f} / 5")
            c3.metric("Taux de TB", f"{d['Taux_TB_Pct']:.1f} %")
            c4.metric("Groupe IA", d['Famille'])

            fig1 = go.Figure()
            fig1.add_trace(go.Contour(x=np.linspace(x_min, x_max, 100), y=np.linspace(y_min, y_max, 100), z=Z_mapped, colorscale=c_scale, showscale=False, opacity=0.25, hoverinfo='skip'))
            fig1.add_trace(go.Scatter(x=stats_all['Note_Moyenne'], y=stats_all['Taux_TB_Pct'], mode='markers', marker=dict(size=6, color=stats_all['Couleur'], opacity=0.6, line=dict(width=0.5, color='black')), text=stats_all['Texte_Survol'], hoverinfo='text', showlegend=False))

            m, b = np.polyfit(stats_all['Note_Moyenne'], stats_all['Taux_TB_Pct'], 1)
            r = np.corrcoef(stats_all['Note_Moyenne'], stats_all['Taux_TB_Pct'])[0, 1]
            fig1.add_trace(go.Scatter(x=np.array([x_min, x_max]), y=m*np.array([x_min, x_max])+b, mode='lines', line=dict(color='black', width=2, dash='solid'), name=f"Tendance (r={r:.2f} | R²={r**2:.2f})", hoverinfo='skip'))
            fig1.add_trace(go.Scatter(x=[d['Note_Moyenne']], y=[d['Taux_TB_Pct']], mode='markers+text', text=[f"<b>{q}</b>"], textposition="top center", marker=dict(size=30, symbol="star", color=d['Couleur'], line=dict(width=2, color="black")), textfont=dict(size=22, color="black", family="Arial Black"), hoverinfo='skip', showlegend=False))
            fig1 = ajouter_lignes_moyennes(fig1)

            fig1.update_layout(height=600, template="plotly_white", plot_bgcolor='white', paper_bgcolor='white', title=dict(text="Position de votre prénom", font=dict(color="black", size=20, family="Arial Black")), xaxis=dict(title=dict(text="Note Moyenne Globale (sur 5)", font=dict(color="black")), tickfont=dict(color="black"), gridcolor='rgba(200, 200, 200, 0.4)'), yaxis=dict(title=dict(text="% Mentions Très Bien & Félicitations", font=dict(color="black")), tickfont=dict(color="black"), gridcolor='rgba(200, 200, 200, 0.4)'), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="black")), hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial"))
            st.plotly_chart(fig1, use_container_width=True, theme=None)

# ==========================================
# ONGLET 2 : VARIANTES ORTHOGRAPHIQUES
# ==========================================
with tab2:
    st.header("L'impact de l'écriture")
    multi = stats_all.groupby('Base')['Prénom usuel'].nunique()
    top_bases = sorted(multi[multi >= 2].index)
    base_q = st.selectbox("Choisissez une racine phonétique :", top_bases)
    df_v = stats_all[stats_all['Base'] == base_q].sort_values('Note_Moyenne')
    
    fig2 = go.Figure()
    fig2.add_trace(go.Contour(x=np.linspace(x_min, x_max, 100), y=np.linspace(y_min, y_max, 100), z=Z_mapped, colorscale=c_scale, showscale=False, opacity=0.25, hoverinfo='skip'))
    fig2.add_trace(go.Scatter(x=stats_all['Note_Moyenne'], y=stats_all['Taux_TB_Pct'], mode='markers', marker=dict(size=6, color=stats_all['Couleur'], opacity=0.4, line=dict(width=0.5, color='black')), text=stats_all['Texte_Survol'], hoverinfo='text', showlegend=False))
    fig2.add_trace(go.Scatter(x=df_v['Note_Moyenne'], y=df_v['Taux_TB_Pct'], mode='lines+markers+text', text=df_v['Prénom usuel'], textposition="top center", line=dict(color='black', dash='dot', width=2), marker=dict(size=18, color=df_v['Couleur'], line=dict(width=2, color="black")), textfont=dict(size=18, family="Arial Black", color="black"), hoverinfo='skip', showlegend=False))
    for i in range(5): fig2.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color=colors_pro[i]), name=labels_pro[i]))
    fig2 = ajouter_lignes_moyennes(fig2)

    fig2.update_layout(height=700, template="plotly_white", plot_bgcolor='white', paper_bgcolor='white', xaxis=dict(title=dict(text="Note Moyenne Globale (sur 5)", font=dict(color="black")), tickfont=dict(color="black"), gridcolor='rgba(200, 200, 200, 0.4)'), yaxis=dict(title=dict(text="% Mentions Très Bien & Félicitations", font=dict(color="black")), tickfont=dict(color="black"), gridcolor='rgba(200, 200, 200, 0.4)'), legend=dict(orientation="h", y=-0.2, font=dict(color="black")))
    st.plotly_chart(fig2, use_container_width=True, theme=None)

# ==========================================
# ONGLET 3 : LES 8 CLASSES DE GENRE
# ==========================================
with tab3:
    st.header("L'impact du Genre (Garçons vs Filles)")
    bins = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
    labels_bins = ["[-1 à -0.75] Très Garçon", "[-0.75 à -0.5] Garçon", "[-0.5 à -0.25] Tendance Garçon", "[-0.25 à 0] Légèrement Garçon", "[0 à 0.25] Légèrement Fille", "[0.25 à 0.5] Tendance Fille", "[0.5 à 0.75] Fille", "[0.75 à 1] Très Fille"]
    
    stats_all['Classe_Genre'] = pd.cut(stats_all['Score_Genre'], bins=bins, labels=labels_bins, include_lowest=True)
    stats_genre = stats_all.groupby('Classe_Genre', observed=False).agg(Moy_X=('Note_Moyenne', 'mean'), Moy_Y=('Taux_TB_Pct', 'mean'), Nb_Prenoms=('Prénom usuel', 'count')).reset_index()
    stats_genre = stats_genre[stats_genre['Nb_Prenoms'] > 0]
    
    fig3 = go.Figure()
    fig3.add_trace(go.Contour(x=np.linspace(x_min, x_max, 100), y=np.linspace(y_min, y_max, 100), z=Z_mapped, colorscale=c_scale, showscale=False, opacity=0.25, hoverinfo='skip'))
    fig3.add_trace(go.Scatter(x=stats_all['Note_Moyenne'], y=stats_all['Taux_TB_Pct'], mode='markers', marker=dict(size=4, color='gray', opacity=0.3), hoverinfo='skip', showlegend=False))
    
    hover_genre = ("<b>" + stats_genre['Classe_Genre'].astype(str) + "</b><br>Prénoms analysés : " + stats_genre['Nb_Prenoms'].astype(str) + "<br>Moyenne : " + stats_genre['Moy_X'].round(2).astype(str) + " / 5<br>Taux TB : " + stats_genre['Moy_Y'].round(1).astype(str) + " %")
    fig3.add_trace(go.Scatter(x=stats_genre['Moy_X'], y=stats_genre['Moy_Y'], mode='lines+markers+text', text=[l.split(']')[1] for l in labels_bins], textposition="top center", line=dict(color='black', width=3), marker=dict(size=22, color=np.linspace(0, 1, len(stats_genre)), colorscale='Bluered', line=dict(width=2, color="black")), textfont=dict(size=14, family="Arial Black", color="black"), hovertext=hover_genre, hoverinfo='text', name="Trajectoire Genre"))
    fig3 = ajouter_lignes_moyennes(fig3)

    fig3.update_layout(height=700, template="plotly_white", plot_bgcolor='white', paper_bgcolor='white', title=dict(text="Évolution des performances académiques", font=dict(color="black", size=18, family="Arial Black")), xaxis=dict(title=dict(text="Note Moyenne", font=dict(color="black")), tickfont=dict(color="black"), gridcolor='rgba(200, 200, 200, 0.4)'), yaxis=dict(title=dict(text="% TB", font=dict(color="black")), tickfont=dict(color="black"), gridcolor='rgba(200, 200, 200, 0.4)'))
    st.plotly_chart(fig3, use_container_width=True, theme=None)

# ==========================================
# ONGLET 4 : ACADÉMIES ET GÉOGRAPHIE
# ==========================================
with tab4:
    st.header("Géographie de l'Excellence")
    st.write("Analyse spatiale : Positionnement des académies et points cardinaux.")

    fig4 = go.Figure()
    fig4.add_trace(go.Contour(x=np.linspace(x_min, x_max, 100), y=np.linspace(y_min, y_max, 100), z=Z_mapped, colorscale=c_scale, showscale=False, opacity=0.25, hoverinfo='skip'))
    fig4.add_trace(go.Scatter(x=stats_all['Note_Moyenne'], y=stats_all['Taux_TB_Pct'], mode='markers', marker=dict(size=4, color='gray', opacity=0.15), hoverinfo='skip', showlegend=False))

    hover_acad = ("<b>" + stats_acad['Académie'] + "</b><br>👥 Candidats : " + stats_acad['Total_Candidats'].astype(str) + "<br>🎓 Note Moy. : " + stats_acad['Note_Moyenne'].round(2).astype(str) + " / 5<br>⭐ Taux TB : " + stats_acad['Taux_TB_Pct'].round(1).astype(str) + " %")
    fig4.add_trace(go.Scatter(x=stats_acad['Note_Moyenne'], y=stats_acad['Taux_TB_Pct'], mode='markers+text', text=stats_acad['Académie'].str.capitalize(), textposition="top center", marker=dict(size=14, color=stats_acad['Couleur'], line=dict(width=1, color="black")), textfont=dict(size=11, family="Arial", color="black"), hovertext=hover_acad, hoverinfo='text', showlegend=False))

    hover_card = ("<b>Direction : " + stats_cardinal['Cardinal'] + "</b><br>👥 Candidats : " + stats_cardinal['Total_Candidats'].astype(str) + "<br>🎓 Note Moy. : " + stats_cardinal['Note_Moyenne'].round(2).astype(str) + " / 5<br>⭐ Taux TB : " + stats_cardinal['Taux_TB_Pct'].round(1).astype(str) + " %")
    fig4.add_trace(go.Scatter(x=stats_cardinal['Note_Moyenne'], y=stats_cardinal['Taux_TB_Pct'], mode='lines+markers+text', text=stats_cardinal['Cardinal'], textposition="bottom center", line=dict(color='black', width=3, dash='dash'), marker=dict(size=28, symbol='diamond', color=stats_cardinal['Couleur'], line=dict(width=3, color="black")), textfont=dict(size=18, family="Arial Black", color="black"), hovertext=hover_card, hoverinfo='text', showlegend=False))
    
    fig4 = ajouter_lignes_moyennes(fig4)

    fig4.update_layout(height=800, template="plotly_white", plot_bgcolor='white', paper_bgcolor='white', title=dict(text="Académies et Points Cardinaux (N/S/E/O)", font=dict(color="black", size=20, family="Arial Black")), xaxis=dict(title=dict(text="Note Moyenne Globale", font=dict(color="black")), tickfont=dict(color="black"), gridcolor='rgba(200, 200, 200, 0.4)'), yaxis=dict(title=dict(text="% Mentions Très Bien", font=dict(color="black")), tickfont=dict(color="black"), gridcolor='rgba(200, 200, 200, 0.4)'))
    st.plotly_chart(fig4, use_container_width=True, theme=None)

# ==========================================
# ONGLET 5 : DOCUMENTATION STATISTIQUE ET SOURCES
# ==========================================
with tab5:
    st.header("📖 Méthodologie, Sources & Limites")
    st.write("Ce document technique détaille les méthodes algorithmiques et les formules mathématiques utilisées dans la classification et la projection des données de cette application.")
    
    st.markdown(r"""
    ### 1. Partitionnement des Données (Algorithme K-Means)
    Afin de segmenter la population étudiante en 5 territoires de réussite ("Performances Faibles" à "Excellence Académique"), nous appliquons l'algorithme d'apprentissage non supervisé de Lloyd ($K$-Means).
    
    **Standardisation ($Z$-Score) :**
    Avant le partitionnement, les variables de performance (Note moyenne et % de mentions Très Bien) subissent une standardisation afin que chaque axe ait un poids équivalent dans le calcul de la distance :
    $$ z_{i} = \frac{x_{i} - \mu}{\sigma} $$
    
    **Fonction objectif :**
    L'algorithme identifie les 5 centroïdes $C = \{\mu_1, \mu_2, ..., \mu_5\}$ en minimisant l'inertie intra-classe (la somme des variances au carré) :
    $$ \arg\min_{S} \sum_{i=1}^{k} \sum_{x \in S_i} || x - \mu_i ||^2 $$

    ### 2. Régression Linéaire (Tendance)
    La droite de tendance visible sur le graphique est calculée via la méthode des moindres carrés ordinaires (MCO), dont l'équation est :
    $$ y = mx + b $$
    
    La pertinence de ce modèle linéaire est évaluée par le **coefficient de corrélation de Pearson ($r$)** et le **coefficient de détermination ($R^2$)** :
    $$ r = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^{n} (y_i - \bar{y})^2}} $$
    *Le $R^2$ indique la part de variance du taux d'excellence qui est strictement expliquée par la note moyenne brute.*

    ### 3. Classification Géospatiale (Points Cardinaux)
    Pour déterminer si un candidat (ou une académie) appartient au Nord, Sud, Est ou Ouest, nous calculons l'angle de sa coordonnée par rapport au barycentre de l'ensemble des candidats.
    
    **Barycentre médian :**
    $$ \tilde{\lambda} = \text{médiane}(\text{Longitudes}), \quad \tilde{\phi} = \text{médiane}(\text{Latitudes}) $$
    
    **Trigonométrie Spatiale :**
    L'angle $\theta$ (en radians) est calculé grâce à la fonction `atan2` :
    $$ \theta = \text{arctan2}(\phi_i - \tilde{\phi}, \lambda_i - \tilde{\lambda}) $$
    Les individus sont ensuite classés selon des secteurs angulaires de $\frac{\pi}{2}$ radians (ex: le Nord correspond à $\theta \in [\frac{\pi}{4}, \frac{3\pi}{4}[$).

    ### 4. Heuristique de Genre
    En l'absence de la variable "Sexe" dans les données ouvertes du Gouvernement, un algorithme d'inférence sémantique est utilisé. Il score le prénom sur un axe de $[-1, 1]$ (de strictement garçon à strictement fille) en analysant les terminaisons phonétiques régulières de la langue française (ex: les suffixes *"-ine"*, *"-elle"* induisent $+0.8$, tandis que *"-us"*, *"-ic"* induisent $-0.8$). Un bruit algorithmique déterministe (modulo de la somme ASCII) est ajouté afin de créer une distribution continue permettant la création de 8 quantiles de trajectoire.
    
    ---
    
    ### 5. Sources, Inspirations & Enseignements
    ⚠️**Avertissement de rigueur :** Ce travail n'est absolument pas parfait : manque de données, biais liés à la récupération, classes inégales. Ceci doit être vu comme un POC (Proof of Concept) étudiant plutôt qu'une véritable ressource scientifique.
    * **Avertissement :** Ce travail n’a en aucun cas pour objectif de stigmatiser qui que ce soit, ni de constituer un document permettant d’établir des classements ou des hiérarchies entre individus, notamment au regard de critères tels que le prénom, l’origine, la localisation, le sexe ou toute autre caractéristique personnelle.
    Les données collectées dans le cadre de cette enquête sont utilisées exclusivement à des fins d’analyse globale et d’amélioration interne. Elles sont traitées de manière anonyme et agrégée, conformément aux principes de minimisation des données et de respect de la vie privée.
    Ce dispositif s’inscrit dans le respect du cadre légal en vigueur, notamment le Règlement Général sur la Protection des Données, ainsi que des principes fondamentaux de non-discrimination et d’égalité de traitement.
    Aucune exploitation individuelle ou nominative ne sera réalisée, et les résultats ne donneront lieu à aucune prise de décision automatisée ou profilage au sens de la réglementation applicable.
    
    La réalisation de ce POC technique a été rendue possible grâce aux enseignements et à la documentation d'experts :
    
    * **Méthodes non-supervisées & Algorithme K-Means :** Enseignements et travaux d'[Ivan Kojadinovic](https://scholar.google.com/citations?user=i9AdNFsAAAAJ&hl=fr), Professeur de statistiques à l'Université de Pau et des Pays de l'Adour (UPPA), chercheur au Laboratoire de Mathématiques et de leurs Applications.
    * **Régression Linéaire & Modélisation :** Enseignements de [Walter TINSSON](https://scholar.google.com/citations?user=YE1UobUAAAAJ&hl=en), Maître de Conférences (section 26) à l'Université de Pau.
    * **Sociologie des Prénoms :** Inspiré par les célèbres travaux de [Baptiste Coulmont](https://scholar.google.com/citations?hl=fr&user=09xFCcMAAAAJ), Professeur de sociologie à l'École normale supérieure Paris-Saclay et expert reconnu dans l'étude sociologique des prénoms au Baccalauréat.
    * **Inspiration & Vulgarisation scientifique :** Les travaux de [Mehdi Moussaïd](https://scholar.google.com/citations?user=7R0KDB0AAAAJ&hl=fr) (Fouloscopie), chercheur au Max Planck Institute for Human Development.
    * **Documentation additionnelle :** Recherches appuyées par Wikipédia et l'assistant IA Google Gemini.

    > 
    """, unsafe_allow_html=True)
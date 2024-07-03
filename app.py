import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Función para generar la nube de palabras
def generate_wordcloud(text, stopwords, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords, collocations=False).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    st.pyplot(plt)

# Función para extraer frases comunes
def extract_common_phrases(text, stopwords):
    words = re.findall(r'\b\w+\b', text.lower())
    words = [word for word in words if word not in stopwords]
    phrases = [' '.join(words[i:i+5]) for i in range(len(words)-4)]  # Extraer frases de longitud 5
    phrase_counts = Counter(phrases)
    common_phrases = [phrase for phrase, count in phrase_counts.most_common(10)]  # Obtener las 10 frases más comunes
    return common_phrases

# Función para clasificar las reseñas basadas en la calificación de estrellas
def classify_reviews(df):
    df['Sentiment'] = df['Star Rating'].apply(lambda x: 'Positive' if x > 3 else 'Negative' if x < 3 else 'Neutral')
    return df

# Función para realizar análisis de temas
def topic_modeling(texts, num_topics=5):
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=0)
    lda.fit(dtm)
    return lda, vectorizer

def display_topics(model, feature_names, no_top_words, texts):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        topic_words = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
        # Crear una oración coherente utilizando las palabras del tema y ejemplos de textos
        example_sentences = [text for text in texts if all(word in text.lower() for word in topic_words)]
        if example_sentences:
            sentence = example_sentences[0]  # Tomar la primera oración que contiene las palabras del tema
            for word in topic_words:
                sentence = sentence.replace(word, f"**{word}**")  # Resaltar las palabras clave
            topics.append(sentence)
        else:
            topics.append(" ".join(topic_words))
    return topics

def main():
    st.title("Generador de nube de palabras y análisis de reseñas")
    uploaded_file = st.file_uploader("Cargame el archivito de CSV dale no tengas miedo:", type="csv")
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.dataframe(data.head())
        
        review_texts = data['Review Text'].dropna().str.cat(sep=' ')
        
        # Lista de stopwords refinada y extendida
        stopwords = set([
            'yo', 'tú', 'vos', 'él', 'ella', 'nosotros', 'nosotras', 'vosotros', 'vosotras', 'ellos', 'ellas', 'me', 'te', 'se', 
            'nos', 'os', 'mi', 'mis', 'tu', 'tus', 'su', 'sus', 'nuestro', 'nuestra', 'nuestros', 'nuestras', 'vuestro', 
            'vuestra', 'vuestros', 'vuestras', 'mío', 'mía', 'míos', 'mías', 'tuyo', 'tuya', 'tuyos', 'tuyas', 'suyo', 
            'suya', 'suyos', 'suyas', 'y', 'o', 'pero', 'aunque', 'sino', 'porque', 'pues', 'como', 'que', 'cuando', 
            'donde', 'si', 'es', 'en', 'de', 'del', 'por', 'con', 'para', 'muy', 'más', 'ya', 'al', 'a', 'también', 
            'entre', 'sin', 'sobre', 'hasta', 'durante', 'tras', 'mientras', 'además', 'entonces', 'luego', 'siempre', 
            'aquí', 'allí', 'ahora', 'después', 'antes', 'ayer', 'hoy', 'mañana', 'siempre', 'nunca', 'tal', 'vez', 
            'solo', 'solamente', 'igual', 'tan', 'tanto', 'como', 'así', 'todavía', 'incluso', 'mientras', 'aunque', 
            'sin', 'embargo', 'no', 'ni', 'sí', 'aun', 'sólo', 'mismo', 'a', 'ante', 'bajo', 'cabe', 'con', 'contra', 
            'de', 'desde', 'durante', 'en', 'entre', 'hacia', 'hasta', 'mediante', 'para', 'por', 'según', 'sin', 'so', 
            'sobre', 'tras', 'versus', 'vía', 'un', 'una', 'unos', 'unas', 'el', 'la', 'los', 'las', 'lo', 'son', 'está', 
            'están', 'estoy', 'estamos', 'estás', 'ser', 'fue', 'fuera', 'fueron', 'siendo', 'serán', 'soy', 'sea', 'sean', 
            'sido', 'siendo', 'tener', 'tengo', 'tiene', 'tienen', 'tenido', 'teniendo'
        ])
        
        # Clasificación de reseñas
        classified_reviews = classify_reviews(data)
        
        if st.button("Generar nube de palabras más positivas"):
            positive_reviews_text = ' '.join(classified_reviews[classified_reviews['Sentiment'] == 'Positive']['Review Text'].dropna())
            generate_wordcloud(positive_reviews_text, stopwords, "Nube de Palabras Positivas")
        
        if st.button("Generar nube de palabras más negativas"):
            negative_reviews_text = ' '.join(classified_reviews[classified_reviews['Sentiment'] == 'Negative']['Review Text'].dropna())
            generate_wordcloud(negative_reviews_text, stopwords, "Nube de Palabras Negativas")
        
        if st.button("Análisis de temas"):
            review_texts_list = data['Review Text'].dropna().tolist()
            lda_model, vectorizer = topic_modeling(review_texts_list)
            topics = display_topics(lda_model, vectorizer.get_feature_names_out(), 10, review_texts_list)
            st.subheader("Temas principales en las reseñas")
            for i, topic in enumerate(topics):
                st.write(f"**Tema {i+1}:** {topic}")
        
        st.subheader("Análisis de distribución de calificaciones")
        ratings_count = data['Star Rating'].value_counts().sort_index()
        fig, ax = plt.subplots()
        ratings_count.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title("Distribución de calificaciones en estrellas (del 1 al 5)")
        ax.set_xlabel("Calificación de estrellas")
        ax.set_ylabel("Cantidad de reseñas")
        st.pyplot(fig)
        
        positive_reviews = classified_reviews[classified_reviews['Sentiment'] == 'Positive'].shape[0]
        negative_reviews = classified_reviews[classified_reviews['Sentiment'] == 'Negative'].shape[0]
        neutral_reviews = classified_reviews[classified_reviews['Sentiment'] == 'Neutral'].shape[0]
        
        st.subheader("Análisis de sentimiento basado en calificación de estrellas")
        st.write(f"Reseñas positivas: {positive_reviews}")
        st.write(f"Reseñas negativas: {negative_reviews}")
        st.write(f"Reseñas neutrales: {neutral_reviews}")
        
        # Gráfico de distribución de reseñas por sentimiento
        sentiment_counts = classified_reviews['Sentiment'].value_counts()
        fig, ax = plt.subplots()
        sentiment_counts.plot(kind='bar', ax=ax, color=['green', 'red', 'blue'])
        ax.set_title("Distribución de sentimiento de reseñas")
        ax.set_xlabel("Sentimiento")
        ax.set_ylabel("Cantidad de reseñas")
        st.pyplot(fig)

if __name__ == "__main__":
    main()

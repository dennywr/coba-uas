import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import re
import requests
from bs4 import BeautifulSoup
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

join = []
temp = []

with st.sidebar:
  selected = option_menu('Klasifikasi Berita', ['Crawling Data', 'Load Data', 'Preprocessing', 'Ekstraksi Fitur', 'Klasifikasi', 'Testing'], default_index=0)
st.title("Klasifikasi Berita")
##### Crwaling Data
def crawlingPta():
  st.subheader("Crawling Portal Berita Antaranews")
  url = st.text_input('Inputkan url mediaindonesi berdasarkan topik di sini', 'https://www.antaranews.com/')
  button = st.button('Crawling')
  if (button):
    # masukkan url
    response = requests.get(url) 
    # Isi teks dari respons HTTP yang diterima dari server web setelah melakukan permintaan GET.
    soup = BeautifulSoup(response.text, 'html.parser') 
    # menemukan semua list yang berisi link kategori
    first_page = soup.findAll('li',"dropdown mega-full menu-color1") 

    # menyimpan kategori
    save_categori = []
    for links in first_page:
      categori = links.find('a').get('href')
      save_categori.append(categori)
    # save_categori

    # categori yang akan disearch terdapat pada indeks 1 (politik)
    categori_search = [save_categori[3],save_categori[6],save_categori[9]] 
    categori_search

    # Inisialisasi list untuk menyimpan data berita
    datas = []

    # Iterasi melalui halaman berita
    for ipages in range(1, 3):

        # Iterasi melalui setiap kategori berita
        for beritas in categori_search:
            # Permintaan untuk halaman berita
            response_berita = requests.get(beritas + "/" + str(ipages))
            namecategori = beritas.split("/")

            # Parsing halaman berita dengan BeautifulSoup
            soup_berita = BeautifulSoup(response_berita.text, 'html.parser')
            pages_berita = soup_berita.findAll('article', {'class': 'simple-post simple-big clearfix'})

            # Iterasi melalui setiap artikel dalam halaman berita
            for items in pages_berita:
                # Mendapatkan link artikel
                get_link_in = items.find("a").get("href")

                # Request untuk halaman artikel
                response_artikel = requests.get(get_link_in)
                soup_artikel = BeautifulSoup(response_artikel.text, 'html.parser')

                # Ekstraksi informasi dari halaman artikel
                judul = soup_artikel.find("h1", "post-title").text if soup_artikel.findAll("h1", "post-title") else ""
                label = namecategori[-1]
                date = soup_artikel.find("span", "article-date").text if soup_artikel.find("span", "article-date") else "Data tanggal tidak ditemukan"

                trash1 = ""
                cek_baca_juga = soup_artikel.findAll("span", "baca-juga")
                if cek_baca_juga:
                    for bacas in cek_baca_juga:
                        text_trash = bacas.text
                        trash1 += text_trash + ' '

                artikels = soup_artikel.find_all('div', {'class': 'post-content clearfix'})
                artikel_content = artikels[0].text if artikels else ""
                artikel = artikel_content.replace("\n", " ").replace("\t", " ").replace("\r", " ").replace(trash1, "").replace("\xa0", "")

                author = soup_artikel.find("p", "text-muted small mt10").text.replace("\t\t", "") if soup_artikel.findAll("p", "text-muted small mt10") else ""

                # Menambahkan data artikel ke dalam list
                datas.append({'Tanggal': date, 'Penulis': author, 'Judul': judul, 'Artikel': artikel, 'Label': label})
    # result = pd.dataFrame(datas)
    st.dataframe(datas)


##### Load Data
def loadData():
  st.subheader("Load Data:")
  data_url = st.text_input('Enter URL of your CSV file here', 'https://raw.githubusercontent.com/dennywr/cobaprosaindata/main/berita_antaranews_fhd.csv')

  @st.cache_resource
  def load_data():
      data = pd.read_csv(data_url, index_col=False)
      # data['nomor\ufeff'] += 1
      return data

  df = load_data()
  df['Penulis'].dropna(inplace=True)
  df['Artikel'].dropna(inplace=True)
  # df.set_index('nomor\ufeff', inplace=True)
  # df.index += 1
  df['Artikel'] = df['Artikel'].fillna('').astype(str)
  # if(selected == 'Load Data'):
  st.dataframe(df)
  return (df['Judul'])
    

##### Preprocessing
def preprocessing():
  st.subheader("Preprocessing:")
  st.text("Menghapus karakter spesial")

  ### hapus karakter spesial
  @st.cache_resource
  def load_data():
      data = pd.read_csv('https://raw.githubusercontent.com/dennywr/cobaprosaindata/main/berita_antaranews_fhd.csv', index_col=False)
      # data['nomor\ufeff'] += 1
      return data

  df = load_data()
  def removeSpecialText (text):
    text = text.replace('\\t',"").replace('\\n',"").replace('\\u',"").replace('\\',"").replace('None',"")
    text = text.encode('ascii', 'replace').decode('ascii')
    return text.replace("http://"," ").replace("https://", " ")
  
  df['Artikel'] = df['Artikel'].astype(str).apply(removeSpecialText)
  df['Artikel'] = df['Artikel'].apply(removeSpecialText)
  # df.index += 1
  st.dataframe(df['Artikel'])
  

  ### hapus tanda baca
  st.text("Menghapus tanda baca")
  def removePunctuation(text):
    text = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text)
    return text

  df['Artikel'] = df['Artikel'].apply(removePunctuation)
  st.dataframe(df['Artikel'])

  ### hapus angka pada teks
  st.text("Menghapus angka pada teks")
  def removeNumbers (text):
    return re.sub(r"\d+", "", text)
  df['Artikel'] = df['Artikel'].apply(removeNumbers)
  st.dataframe(df['Artikel'])

  ### case folding
  st.text("Mengubah semua huruf pada teks menjadi huruf kecil")
  def casefolding(Comment):
    Comment = Comment.lower()
    return Comment
  df['Artikel'] = df['Artikel'].apply(casefolding)
  st.dataframe(df['Artikel'])
  
  st.text("Menghapus stopwords")
  def removeStopwords(text):
    stop_words = set(stopwords.words('indonesian'))
    words = nltk.word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)
  df['Artikel'] = df['Artikel'].apply(removeStopwords)
  st.dataframe(df['Artikel'])


def preprocessingTanpaOutput():

  ### hapus karakter spesial
  @st.cache_resource
  def load_data():
      data = pd.read_csv('https://raw.githubusercontent.com/dennywr/cobaprosaindata/main/berita_antaranews_fhd.csv')
      return data

  df = load_data()
  ### if(hapusKarakterSpesial):
  def removeSpecialText (text):
    text = text.replace('\\t',"").replace('\\n',"").replace('\\u',"").replace('\\',"").replace('None',"")
    text = text.encode('ascii', 'replace').decode('ascii')
    return text.replace("http://"," ").replace("https://", " ")
  
  df['Artikel'] = df['Artikel'].astype(str).apply(removeSpecialText)
  df['Artikel'] = df['Artikel'].apply(removeSpecialText)
  df['Artikel'].dropna(inplace=True)


  # hapusTandaBaca = st.button("Hapus Tanda Baca")
  def removePunctuation(text):
    text = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text)
    return text

  df['Artikel'] = df['Artikel'].apply(removePunctuation)

  ### hapus angka pada teks
  def removeNumbers (text):
    return re.sub(r"\d+", "", text)
  df['Artikel'] = df['Artikel'].apply(removeNumbers)

  ### case folding
  def casefolding(Comment):
    Comment = Comment.lower()
    return Comment
  df['Artikel'] = df['Artikel'].apply(casefolding)

  ### stopwords removal
  def removeStopwords(text):
    stop_words = set(stopwords.words('indonesian'))
    words = nltk.word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)
  df['Artikel'] = df['Artikel'].apply(removeStopwords)

  # ### stemming
  # def stemText(text):
  #   factory = StemmerFactory()
  #   stemmer = factory.create_stemmer()
  #   return stemmer.stem(text)
  # df['Artikel'] = df['Artikel'].apply(stemText)

  return (df["Artikel"], df["Judul"], df["Label"])


##### Ekstraksi Fitur
def ekstraksiFitur():
  import nltk
  from nltk.tokenize import RegexpTokenizer
  from sklearn.decomposition import TruncatedSVD
  from sklearn.feature_extraction.text import TfidfVectorizer
  from nltk.corpus import stopwords

  nltk.download('stopwords', quiet=True)

  st.subheader("Ekstraksi Fitur (TF-IDF):")
  stopwords = stopwords.words('indonesian')

  from sklearn.feature_extraction.text import CountVectorizer

  coun_vect = CountVectorizer(stop_words=stopwords)
  # coun_vect = CountVectorizer()
  count_matrix = coun_vect.fit_transform(preprocessingTanpaOutput()[0])
  count_array = count_matrix.toarray()

  df = pd.DataFrame(data=count_array, columns=coun_vect.vocabulary_.keys())

  # Menampilkan DataFrame menggunakan streamlit
  st.text(ekstraksiFiturTanpaOutput()[0].shape)

  df = pd.concat([preprocessingTanpaOutput()[1], df], axis=1)

  st.dataframe(df)

  tokenizer = RegexpTokenizer(r'\w+')
  vectorizer = TfidfVectorizer(lowercase=True,
                          stop_words=stopwords,
                          tokenizer = tokenizer.tokenize)

  tfidf_matrix = vectorizer.fit_transform(preprocessingTanpaOutput()[0])
  tfidf_terms = vectorizer.get_feature_names_out()
  # st.text(tfidf_matrix)
  vsc = pd.DataFrame(data=tfidf_matrix.toarray(),columns = vectorizer.vocabulary_.keys())
  vsc = pd.concat([preprocessingTanpaOutput()[1], vsc], axis=1)
  st.text("Vector Space Model")
  st.dataframe(vsc)


##### Ekstraksi Fitur
def ekstraksiFiturTanpaOutput():
  import nltk
  from nltk.tokenize import RegexpTokenizer
  from sklearn.decomposition import TruncatedSVD
  from sklearn.feature_extraction.text import TfidfVectorizer
  from nltk.corpus import stopwords

  nltk.download('stopwords', quiet=True)

  stopwords = stopwords.words('indonesian')

  tokenizer = RegexpTokenizer(r'\w+')
  vectorizer = TfidfVectorizer(lowercase=True,
                          stop_words=stopwords,
                          tokenizer = tokenizer.tokenize)

  tfidf_matrix = vectorizer.fit_transform(preprocessingTanpaOutput()[0])
  tfidf_terms = vectorizer.get_feature_names_out()
  return [tfidf_matrix, tfidf_terms, vectorizer]

# def pca():
#    m

def nb_classifier(show_output=False):
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import classification_report, accuracy_score
  from sklearn.naive_bayes import MultinomialNB
  from sklearn.decomposition import PCA
  from sklearn.preprocessing import MinMaxScaler

  if show_output:
    st.subheader("Klasifikasi (Naive Bayes):")
  # Label
  X = ekstraksiFiturTanpaOutput()[0]
  y = preprocessingTanpaOutput()[2]

  # Bagi dataset menjadi data latih dan data uji
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Assuming X_train.toarray() has shape (number_of_samples, number_of_features)
  valid_n_components = min(X_train.shape[0], X_train.shape[1])

  # Use a value within the valid range for n_components
  n_components = min(100, valid_n_components)
  pca = PCA(n_components=n_components)
  X_train_pca = pca.fit_transform(X_train.toarray())
  X_test_pca = pca.transform(X_test.toarray())

  scaler = MinMaxScaler()
  X_train_pca = scaler.fit_transform(X_train_pca)
  X_test_pca = scaler.transform(X_test_pca)

  # Buat model Naive Bayes
  naive_bayes_model = MultinomialNB()

  # Latih model pada data latih
  naive_bayes_model.fit(X_train_pca, y_train.values.ravel())

  # Lakukan prediksi pada data uji
  y_pred_nb = naive_bayes_model.predict(X_test_pca)
  # Evaluasi performa model Naive Bayes
  if show_output:
    st.text(f'Akurasi: {accuracy_score(y_test, y_pred_nb)}')
    st.text(classification_report(y_test, y_pred_nb))
  return [pca, naive_bayes_model]

def svm_classifier(show_output=False):
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import classification_report, accuracy_score
  from sklearn.decomposition import PCA
  from sklearn.svm import SVC
  from sklearn.preprocessing import MinMaxScaler

  if show_output:
    st.subheader("Klasifikasi (SVM):")
  # Label
  X = ekstraksiFiturTanpaOutput()[0]
  y = preprocessingTanpaOutput()[2]

  # Bagi dataset menjadi data latih dan data uji
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Assuming X_train.toarray() has shape (number_of_samples, number_of_features)
  valid_n_components = min(X_train.shape[0], X_train.shape[1])

  # Use a value within the valid range for n_components
  n_components = min(100, valid_n_components)
  pca = PCA(n_components=n_components)
  X_train_pca = pca.fit_transform(X_train.toarray())
  X_test_pca = pca.transform(X_test.toarray())

  scaler = MinMaxScaler()
  X_train_pca = scaler.fit_transform(X_train_pca)
  X_test_pca = scaler.transform(X_test_pca)

  # Buat model Naive Bayes
  svm_model = SVC(kernel='rbf', C=1, gamma=0.1)

  # Latih model pada data latih
  svm_model.fit(X_train_pca, y_train.values.ravel())

  # Lakukan prediksi pada data uji
  y_pred_svm = svm_model.predict(X_test_pca)
  # Evaluasi performa model Naive Bayes
  if show_output:
    st.text(f'Akurasi: {accuracy_score(y_test, y_pred_svm)}')
    st.text(classification_report(y_test, y_pred_svm))
  return [pca, svm_model]

def testing():
  st.subheader("Prediksi kategori berita (ekonomi, olahraga, hiburan):")
  X_new = st.text_area("Masukkan kalimat di sini", height=250)
  predictButton = st.button('Prediksi')
  if(predictButton):
    X_new_tfidf = ekstraksiFiturTanpaOutput()[2].transform([X_new])

    X_new_pca = svm_classifier(False)[0].transform(X_new_tfidf.toarray())

    predict = svm_classifier(False)[1].predict(X_new_pca)
    st.text(f'Berita tersebut termasuk ke kategori: {predict[0]}')

def main():
  if(selected == 'Crawling Data'):
     crawlingPta()
  elif(selected == 'Load Data'):
     loadData()
  elif(selected == 'Preprocessing'):
     preprocessing()
  elif(selected == 'Ekstraksi Fitur'):
     ekstraksiFitur()
  elif(selected == 'Klasifikasi'):
      svm_classifier(True)
  else:
     testing()




if __name__ == "__main__":
    main()


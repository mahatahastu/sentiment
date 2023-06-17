import streamlit as st
import pandas as pd
import altair as alt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import os
import pickle





def main():
    st.title("Data Stemming")
    
    # Read the CSV file if it exists
    csv_path = "stemmed_data.csv"
      
    # Check if 'stemmed.csv' file exists
    if not os.path.exists(csv_path):
        st.warning("Proses preprocessing dulu")
    else:
        # Read the CSV file
        data = pd.read_csv(csv_path)

        # Remove empty 'teks' column
        data = data.dropna(subset=['stemmed_text'])
        st.dataframe(data)
        st.write(f"Number of rows: {data.shape[0]}")


        
        if st.button("Pembobotan"):
            # Memisahkan fitur (teks tweet) dan label (sentimen)
            X = data['stemmed_text']
            y = data['label']
            # Membagi data menjadi data latih / train dan data uji / test = 0.2
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            # Menggunakan TF-IDF untuk mengekstraksi fitur dari teks tweet
            vectorizer = TfidfVectorizer()
            X_train = vectorizer.fit_transform(X_train)
            X_test = vectorizer.transform(X_test)
            with open('tfidf_vectorizer.pkl', 'wb') as file:
                pickle.dump(vectorizer, file)
            
            # Melatih model SVM
            svm_model = svm.SVC(kernel='linear')
            svm_model.fit(X_train, y_train)
            with open('svm_model.pkl', 'wb') as file:
                pickle.dump(svm_model, file)

            #tampilan file pickle berhasil diinputkan
            processed_filename1 = "tfidf_vectorizer.pkl"
            st.success(f"Data saved as {processed_filename1}")
            processed_filename2 = "svm_model.pkl"
            st.success(f"Data saved as {processed_filename2}")

            # Memprediksi sentimen pada data uji
            y_pred = svm_model.predict(X_test)

            #melihat accuracy model yang dihasilkan
            st.write(classification_report(y_test, y_pred))
            

            
            tab1, tab2 = st.tabs(["Teks", "Label"])

            with tab1:
                col1, col2=st.columns(2)
                with col1:
                    st.info("Teks Training")
                    st.write(X_train)
                with col2:
                    st.info("Teks Uji")    
                    st.write(X_test)

                
            with tab2:
                col3, col4=st.columns(2)
                with col3:
                    st.info("Label Training")    
                    st.write(y_test)
                    st.write(f"Jumlah Data: {y_test.shape[0]}")
                with col4:
                    st.info("Label Uji")   
                    st.write(y_train)
                    st.write(f"Jumlah Data: {y_train.shape[0]}")
                    
if __name__ == '__main__':
    main()
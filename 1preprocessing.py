import streamlit as st

import pandas as pd
import altair as alt
import re



from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

def hapus_teks(text):
    # Mencari dan mengganti "banksyariahindonesia" dengan string kosong
    text_clean = text.replace("Bank Syariah Indonesia", "")
    text_clean = text.replace("Bank", "")
    text_clean = text.replace("Syariah", "")
    text_clean = text.replace("Indonesia", "")
    return text_clean


def clean_text(text):
    
    # Menghapus spasi berlebihan
    cleaned_text = re.sub(r"\s+", " ", text)
    # Mengubah teks menjadi lowercase
    cleaned_text = cleaned_text.lower()
    return cleaned_text

def clean_hastag(text):
    
    # Fungsi untuk menghapus hashtag dari teks  
    cleaned_pagar = re.sub(r'#\w+', '', text)
    return cleaned_pagar

def hapus_angka(teks):
    teks_tanpa_angka = re.sub(r'\d+', '', teks)
    return teks_tanpa_angka

def remove_links(text):
    # Menghapus link dari teks
    cleaned_link = re.sub(r"http\S+|www\S+|https\S+", "", text)
    return cleaned_link

def remove_symbols(text):
    # Menghapus simbol dari teks
    cleaned_symbols = re.sub(r"[^\w\s]", "", text)
    return cleaned_symbols

# Fungsi untuk melakukan stemming pada teks
def stemming_text(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stemmed_text = stemmer.stem(text)
    return stemmed_text


def main():
       
    st.title("Input Data")

    # Menampilkan widget untuk memilih file CSV
    csv_file = st.file_uploader("Choose a CSV file", type="csv")

    if csv_file is not None:
        
        # Membaca file CSV menjadi DataFrame
        dataframe = pd.read_csv(csv_file)
    
        
        st.subheader("Original Data")
        # Menghapus kolom username
        
        # Menampilkan data awal sebelum proses cleansing
        st.dataframe(dataframe)
        st.write(f"Jumlah Data: {dataframe.shape[0]}")
        
        
        # Create Altair chart
        chart = alt.Chart(dataframe).mark_bar().encode(
            x=alt.X('label:N', title='Sentiment Label'),
            y='count()',
            color='label:N',
            tooltip=['label']
        ).interactive()
        # Display the chart using st.altair_chart()
        st.altair_chart(chart, use_container_width=True)


        
        
        # Menampilkan pilihan untuk melakukan cleansing (cleaned data)
        if st.checkbox("Perform Data Cleansing"):
            # Melakukan pemanggilan proses cleansing
            dataframe['clean_text'] = dataframe['full_text'].apply(lambda x: clean_text(hapus_angka(remove_links(remove_symbols(clean_hastag(hapus_teks(x)))))))
            dataframe.drop_duplicates(subset = "clean_text", keep = 'first', inplace = True)

            # Melakukan pembersihan pada kolom teks
            
            
            st.subheader("Cleaned Data")
            # Menampilkan data setelah proses cleansing
            st.dataframe(dataframe)
            st.write(f"Jumlah Data: {dataframe.shape[0]}")
            
            
            # Create Altair chart
            chart = alt.Chart(dataframe).mark_bar().encode(
                x=alt.X('label:N', title='Sentiment Label'),
                y='count()',
                color='label:N',
                tooltip=['label']
            ).interactive()
            # Display the chart using st.altair_chart()
            st.altair_chart(chart, use_container_width=True)
            
            
            if st.button("Stemmed"):
                # Melakukan stemming pada teks dalam kolom 'text'
                dataframe['stemmed_text'] = dataframe['clean_text'].apply(stemming_text)
                # Melakukan pembersihan pada kolom teks
                #dataframe.drop(dataframe.columns[[1]], axis = 1, inplace = True)
                st.subheader("Data Setelah Stemming")
                st.dataframe(dataframe[['label', 'stemmed_text']])
                st.write(f"Jumlah Data: {dataframe.shape[0]}")
            
            
                # Create Altair chart
                chart = alt.Chart(dataframe).mark_bar().encode(
                    x=alt.X('label:N', title='Sentiment Label'),
                    y='count()',
                    color='label:N',
                    tooltip=['label']
                ).interactive()
                # Display the chart using st.altair_chart()
                st.altair_chart(chart, use_container_width=True)
            
            
                processed_filename = "stemmed_data.csv"
                dataframe.to_csv(processed_filename, index=False)
                st.success(f"Data saved as {processed_filename}")
                
if __name__ == '__main__':
    main()


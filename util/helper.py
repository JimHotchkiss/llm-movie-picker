import pandas as pd 

def load_data(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        columns = {
            "type": df["type"],
            "title": df["title"],
            "director": df["director"],
            "cast": df["cast"].tolist(),
            "country": df["country"],
            "date_added": df["date_added"],
            "release_year": df["release_year"],
            "rating": df["rating"],
            "duration": df["duration"],
            "genre": df["listed_in"].tolist(),
            "description": df["description"]
        }
        df_renamed = df.rename(columns={"listed_in": "genre"})
        return df_renamed
    
    elif uploaded_file.name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")
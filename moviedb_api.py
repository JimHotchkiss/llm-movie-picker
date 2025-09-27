import requests 

# MOVIE_API_KEY = "178f905339c28367429420feff5fa554"
URL = f"https://api.themoviedb.org/3/discover/movie?include_adult=false&include_video=false&language=en-US&page=1&sort_by=popularity.desc"

url = "https://api.themoviedb.org/3/authentication"

NOW_PLAYING_URL = "https://api.themoviedb.org/3/movie/now_playing?language=en-US"
# &page=1
POPULAR_URL = "https://api.themoviedb.org/3/movie/popular?language=en-US&page=1"

headers = {
    "accept": "application/json",
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIxNzhmOTA1MzM5YzI4MzY3NDI5NDIwZmVmZjVmYTU1NCIsIm5iZiI6MTU3NjY5OTAyNy45MjYsInN1YiI6IjVkZmE4NDkzMjZkYWMxMDAxMjU5NDdmYSIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.0at98L4tAq7v2SFziYHZjth8dgrJg8QuqfA8mUmnoYA"
}

def loop_data(data):
    for key in data:
        print(f"length: {len(data)}")
        print(f"key: {key}")




try:
    response = requests.get(NOW_PLAYING_URL, headers=headers)
    response.raise_for_status() # Raise an exception for bad status code

    # Parse the JSON response
    movie_data = response.json()
    movie_results = movie_data['results']
    print(f"movie_data: {movie_data}")
    loop_data(movie_results)

except requests.exceptions.RequestException as e:
    print(f"An error occured during the API request: {e}")



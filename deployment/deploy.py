import streamlit as st
import numpy as np
import lightgbm as lgb
import joblib
import ast
import pickle
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import math
import requests
from sklearn.cluster import KMeans
from PIL import Image
from io import BytesIO
import webbrowser
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import GenerationConfig
import os
import torch
import pandas as pd

def convert_date(date_str):
    # Check if the string is in the format 'yyyy-mm'
    if len(date_str) == 7 and date_str[4] == '-':
        #yy-mm only in csv it's found the day to be 01
        return pd.to_datetime(date_str[:4] + '-' + date_str[5:] + '-01', format='%Y-%m-%d')
    # Check if the string is in the format 'yyyy-mm-dd'
    elif len(date_str) == 10 and date_str[4] == '-' and date_str[7] == '-':
        return pd.to_datetime(date_str, format='%Y-%m-%d')
    # Check if the string is in the format 'm/d/yyyy'
    elif '/' in date_str:
        return pd.to_datetime(date_str, format='%m/%d/%Y')
    elif len(date_str) == 4:  # Check if the string is only the year
        return pd.to_datetime(date_str + '-01-01', format='%Y-%m-%d')  
    else:
        return pd.to_datetime(date_str, errors='coerce')






# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if torch.cuda.is_available():
#     print("GPU is available!")
# else:
#     print("GPU is NOT available.")
# model_name='google/flan-t5-xl'

# model_prompt = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
# model_prompt.to(device)

# def make_prompt(examples, testEx):
#     prompt = ""
#     for i in range(len(examples)):
#         prompt += f"""(be strict) Song Name : {examples[i][0]}\n\nIs it Catchy ? {examples[i][1]}\n\n"""
#     prompt += f"""Song Name : {testEx}\n\nIs it Catchy ?"""
#     return prompt

# def isCatchy(testEX,model):
#   examples = [
#     ["I Need You", "Catchy"],
#     ["Hurt", "Catchy"],
#     ["You Take My Breath Away - Mono Version", "Not Catchy"],
#     ["If I Give My Heart to You (with The Mellomen)", "Not Catchy"],
#     ["Throwing It All Away - 2007 Remaster", "Not Catchy"],
#     ["Does Your Chewing Gum Lose Its Flavour (On The Bedpost Overnight)", "Not Catchy"],
#     ["The Rock And Roll Waltz", "Not Catchy"],]
#   one_shot_prompt = make_prompt(examples, testEX)

#   inputs = tokenizer(one_shot_prompt, return_tensors='pt').to(device)
#   output = tokenizer.decode(
#         model.generate(
#             inputs["input_ids"],
#             max_new_tokens=200,
#         )[0],
#         skip_special_tokens=True
#     )
#   return output
def Generate_description(image_url):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    raw_image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs)
    desc = processor.decode(out[0], skip_special_tokens=True)
    return desc

# Example usage:
# image_url = "https://example.com/your-image.jpg"

def Clusters(gen_count,gen_weight,gen_total_weight,day,month,year):
   X = np.array([[gen_count,gen_weight,gen_total_weight,day,month,year]])
   print("Da shape al dakhl ll model 3shan zah2t ", X.shape)
   with open('Kmean.pkl', 'rb') as f:
      model = pickle.load(f)

   cluster_no=model.predict(X)[0]
   print("This is Cluster : ",cluster_no)
   return cluster_no
   

def safe_log(x):
  return np.log(x+1e-8)



def predict(genres, Song_Name, Artist_Name, Album_Name,Date,song_duration,Acoustincess,Danceability,Energy,Instrumentalness,Liveness,Loudness,Speechness,Tempo,Valence,Key,TimeSignature,mode,Song_img_URL):
    print(Date)
    # Get the number of genres entered
    num_entered_genres = 0
   # Convert the string back to a list using ast.literal_eval
    list_from_string = ast.literal_eval(genres)
#["vo","kk"]
    
   
       
    date_right_format= convert_date(Date)
    
    month =date_right_format.month
    day =date_right_format.day
    Year = date_right_format.year
    print(month)
    print(day)
    print(Year)

    is_catchy = 0
    description = Generate_description(Song_img_URL)
    words = description.split()
    num_words_description = len(words)
    def process_image(url):
        try:
            # Request image from URL
            response = requests.get(url)
            image = Image.open(BytesIO(response.content))

            # Convert image to numpy array
            image_array = np.array(image)

            # Calculate standard deviation of RGB values
            std_dev = np.std(image_array, axis=(0, 1)).mean()

            # Define a threshold for colorfulness
            color_threshold = 50.0  # 81 is very colorful

            # Determine if the image is colorful based on standard deviation
            is_colorful_image = std_dev > color_threshold

            # Flatten the array to get all RGB values
            flattened = image_array.reshape(-1, 3)

            # Get unique RGB values and their counts
            unique_colors, counts = np.unique(flattened, axis=0, return_counts=True)

            # Sort colors by frequency (most common colors first)
            sorted_colors = sorted(zip(unique_colors, counts), key=lambda x: -x[1])

            # Return the top color values (RGB) and their frequencies
            top_color_values = sorted_colors[:5]

            # # Open the URL in the default web browser
            # webbrowser.open(url)

            return is_colorful_image, top_color_values

        except Exception as e:
            print("Error processing image:", e)
            return None, None

    is_colorful, colors = process_image(Song_img_URL)
    print("Is colorful:", is_colorful)
    print("Top colors:", colors)

    
   
    #No of words
    text = Song_Name.split()
    no_of_words = len(text)
    #No of chars
    no_of_chars = len(Song_Name)
    
    decade ={"Decade_new millennium":0,"Decade_Thirties":0,"Decade_Sixties":0,"Nighnties":0,"Decade_Seventies":0,"Decade_Eighties":0,"Decade_Fourties":0,"Decade_Fifties":0,"Dcade_old":0}
    weight = 0
    genre_weights = {}
    genre_dic = {}
    try:
     with open('data_for_nour.txt', 'r') as file:
      genre_weights = eval(file.read())
    except (FileNotFoundError, SyntaxError) as e:
      print(f"Error loading genre weights: {e}")

    genre_dic=genre_weights.copy()
    for genre,val in genre_dic.items():
       genre_dic[genre]=0

    total_weight = 0
    flag = False
    not_found = []
#mhataga num of entered genres

#["Pop rock","jazz","pop rock jannah","jannah","nour"]
    for i in list_from_string:
     flag = False
     num_entered_genres = len(i.split())
     for genre in genre_weights:
        genre_words = genre.split()
        if len(genre_words) == num_entered_genres:
            entered_genres_sorted = sorted(genres)
            # Sort the genre words alphabetically
            genre_words_sorted = sorted(genre)
            # Check if the sorted entered genres match the sorted genre words
            if entered_genres_sorted == genre_words_sorted:
                genre_dic[genre]=1
                total_weight = genre_weights[genre]
                # flag = True
                break
            if entered_genres_sorted != genre_words_sorted :
               not_found.append(i)

               
    for j in not_found:
        entered_genres = j.split()
        # for genre in entered_genres:
        #  if genre in genre_weights:
        #      genre_dic[genre]=1
        #      total_weight += genre_weights[genre]
        #  else:
        total_weight += 1
   
    num_entered_genres = len(list_from_string)
    print("The num genre : ",num_entered_genres)
    print("The total weight : ",total_weight)

    #Decade
    if Year < 1920:
     decade["Dcade_old"]=1
    elif Year >= 1930 and Year <1940:
      decade["Decade_Thirties"]=1
    elif Year >= 1940 and Year <1950:
      decade["Decade_Fourties"]=1
    elif Year >= 1950 and Year < 1960:
      decade["Decade_Fifties"]=1
    elif Year >= 1960 and Year<1970:
     decade["Decade_Sixties"]=1
    elif Year >= 1970 and Year <1980:
     decade["Decade_Seventies"]=1
    elif Year >=1980 and Year <1990:
     decade["Decade_Eighties"]=1
    elif Year >= 1990 and Year < 2000:
     decade["Nighnties"]=1
    elif Year >= 2000:
     decade["Decade_new millennium"]=1


#Season
    seasons = {"spring":0,"Autumn":0,"Winter":0,"Summer":0}
    if 3 <= month <= 5:
        seasons["spring"]=1
    elif 6 <= month <= 8:
       seasons["Summer"]=1
    elif 9 <= month <= 11:
        seasons["Autumn"]=1
    else:
        seasons["Winter"]=1

    #Song duration
    large_threshold = 321159.2929926557
    small_threshold = 128093.77759367932

    if song_duration >= large_threshold:
        duration_Category = "large_time"
    elif song_duration >= small_threshold:
        duration_Category = "average_time"
    else:
        duration_Category  = "small_time"

    #Formatted Date
    formatted_date = f"{day}/{month}/{Year}"
   
    # Top 1000 Artist Scraping
    Top_100_art_sc = 0
    # Top 1800 Album
    Top_1800_Album = 0
    #Has potential 
    has_potential =0
    #local
    is_local =0
    
    try:
        with open('col.csv', encoding='iso-8859-1') as file:
            header = next(file)  # Skip the header line
            for line in file:
                try:
                    # Split the line into columns (song_name, rank, year_rank, islocal, num_of_available_market)
                    song_data = line.strip().split(',', 4)
                    song_name_csv, rank, year_rank, islocal, num_of_available_market = map(str.strip, song_data)

                    if song_name_csv.lower() == Song_Name.strip().lower():
                        if int(rank) >= 50:
                            has_potential = 1  # Assuming you still want to use this flag
                        is_local=1
                        break # Stop searching after finding the song

                except ValueError:
                    print("Skipping line due to unexpected format:", line)

    except FileNotFoundError:
        print("File not found. Please check the file path.")


   
    print("Before")
    with open('artist_rankings.csv', 'r') as file:
        for line in file:
            data = line.strip().split(',')
            if len(data) == 2:
                if data[1] == Artist_Name:
                    Top_100_art_sc = 1
                    print("hello")
                    break
    print("After")
    with open('Album_Rank.csv', 'r') as file:
        for line in file:
            data = line.strip()
            if data == Album_Name:
                Top_1800_Album = 1
                print("hello")
                break
    clust_no = Clusters(num_entered_genres,total_weight,num_entered_genres*total_weight,day,month,Year)
    X_regression = np.array([[rank,safe_log(song_duration),math.sqrt(Acoustincess),Danceability,Energy,safe_log(Instrumentalness),safe_log(Liveness),Loudness,safe_log(Speechness),Tempo,Valence,Key,TimeSignature,safe_log(no_of_words),safe_log(no_of_chars),num_of_available_market,year_rank,safe_log(num_entered_genres),safe_log(total_weight),safe_log(num_entered_genres*total_weight),month,Year,int(is_colorful),has_potential,Top_100_art_sc,mode,Top_1800_Album,decade["Decade_Eighties"],decade["Decade_Fifties"],decade["Decade_Fourties"],decade["Dcade_old"],decade["Decade_Seventies"],decade["Decade_Sixties"],decade["Decade_Thirties"],decade["Decade_new millennium"],seasons["Autumn"],seasons["spring"],seasons["Winter"],int(duration_Category=="average_time"),int(duration_Category=="small_time"),
                  genre_dic['country road'],genre_dic["contemporary country"],genre_dic["country dawn"],genre_dic["country"],genre_dic["contemporary r&b"],genre_dic["hip pop"],genre_dic["r&b"],genre_dic["urban contemporary"],genre_dic["bubblegum pop"],genre_dic["movie tunes"],genre_dic["easy listening"],genre_dic["vocal jazz"],genre_dic["adult standards"],genre_dic["lounge"],genre_dic["karaoke"],genre_dic["rock"],genre_dic["glam metal"],genre_dic["hard rock"],genre_dic["album rock"],genre_dic["pop rock"],genre_dic["canadian singer-songwriter"],genre_dic["neo mellow"],genre_dic["lilith"],genre_dic["canadian pop"],genre_dic["singer-songwriter"],genre_dic["neon pop punk"],genre_dic["pop punk"],genre_dic['new wave pop'],genre_dic["pop"],genre_dic['"womens music"'],genre_dic["soft rock"],genre_dic["art rock"],genre_dic["classic rock"],genre_dic["progressive rock"],genre_dic["mellow gold"],genre_dic["symphonic rock"],genre_dic["northern soul"],genre_dic["skiffle"],
                  genre_dic["music hall"], genre_dic["country rock"], genre_dic["folk rock"], genre_dic["classic garage rock"], genre_dic["sunshine pop"], genre_dic["southern rock"], genre_dic["dance pop"], genre_dic["alt z"], genre_dic["rock-and-roll"], genre_dic["hip house"], genre_dic["diva house"],genre_dic["eurodance"], genre_dic["german techno"], genre_dic["torch song"], genre_dic["rockabilly"], genre_dic["melodic rap"], genre_dic["doo-wop"], genre_dic["swamp rock"], genre_dic["neo soul"], genre_dic["blues"], genre_dic["soul"], genre_dic["blues rock"], genre_dic["metal"], genre_dic["alternative metal"], genre_dic["birmingham metal"], genre_dic["nu metal"], genre_dic["nederpop"], genre_dic["classic soul"], genre_dic["motown"], genre_dic["quiet storm"],genre_dic["swing italiano"], genre_dic["hip hop"], genre_dic["canadian hip hop"], genre_dic["rap"], genre_dic["toronto rap"], genre_dic["american folk revival"], genre_dic["folk"], genre_dic["flute rock"], genre_dic["british invasion"], genre_dic["heartland rock"], genre_dic["post-teen pop"], genre_dic["nashville sound"], genre_dic["classic country pop"], genre_dic["east coast hip hop"], genre_dic["hardcore hip hop"], genre_dic["pop rap"], genre_dic["classic girl group"], genre_dic["pop soul"], genre_dic["british blues"], genre_dic["conscious hip hop"], genre_dic["west coast rap"], genre_dic["boy band"], genre_dic["ambient house"], genre_dic["acid house"], genre_dic["big beat"], genre_dic["glam punk"], genre_dic["new jack swing"],
                  genre_dic["atl hip hop"], genre_dic["old school atlanta hip hop"], genre_dic["operatic pop"], genre_dic["deep talent show"], genre_dic["deep adult standards"], genre_dic["merseybeat"], genre_dic["dirty south rap"], genre_dic["trap"], genre_dic["southern hip hop"], genre_dic["wrestling"], genre_dic["new orleans rap"], genre_dic["psychedelic rock"], genre_dic["yacht rock"], genre_dic["piano rock"], genre_dic["sophisti-pop"], genre_dic["chicago rap"], genre_dic["viral trap"], genre_dic["talent show"], genre_dic["trip hop"], genre_dic["beatlesque"], genre_dic["smooth jazz"], genre_dic["smooth saxophone"], genre_dic["disco"], genre_dic["new romantic"], genre_dic["new wave"], genre_dic["permanent wave"], genre_dic["dance rock"], genre_dic["modern rock"], genre_dic["pov: indie"],
                  genre_dic["vocal harmony group"], genre_dic["texas latin rap"], genre_dic["latin hip hop"], genre_dic["chicano rap"], genre_dic["austin singer-songwriter"], genre_dic["ectofolk"], genre_dic["post-disco"], genre_dic["philly soul"], genre_dic["glam rock"], genre_dic["protopunk"], genre_dic["detroit rock"], genre_dic["otacore"], genre_dic["girl group"], genre_dic["vocal house"], genre_dic["funk"], genre_dic["barbadian pop"], genre_dic["miami hip hop"], genre_dic["funk rock"], genre_dic["synth funk"], genre_dic["minneapolis sound"], genre_dic["space age pop"], genre_dic["southern soul"], genre_dic["gangster rap"], genre_dic["persian pop"], genre_dic["dutch prog"], genre_dic["uk post-punk"], genre_dic["classic oklahoma country"],genre_dic["freestyle"], genre_dic["seattle hip hop"], genre_dic["jazz rock"], genre_dic["candy pop"], genre_dic["post-grunge"], genre_dic["instrumental funk"], genre_dic["memphis soul"], genre_dic["instrumental soul"], genre_dic["electric blues"], genre_dic["surf music"], genre_dic["g funk"], genre_dic["uk pop"], genre_dic["metropopolis"], genre_dic["indietronica"], genre_dic["jazz trumpet"], genre_dic["synthpop"], genre_dic["crunk"], genre_dic["military cadence"], genre_dic["italian disco"], genre_dic["p funk"], genre_dic["modern country rock"], genre_dic["detroit hip hop"], genre_dic["queens hip hop"], genre_dic["new orleans soul"], genre_dic["nyc rap"], genre_dic["canadian contemporary r&b"],genre_dic["europop"], genre_dic["bubblegum dance"], genre_dic["bronx hip hop"], genre_dic["novelty"], genre_dic["hi-nrg"], genre_dic["power pop"], genre_dic["tropical house"], genre_dic["electropop"], genre_dic["pixie"], genre_dic["pop emo"], genre_dic["irish rock"], genre_dic["psychedelic soul"], genre_dic["rock keyboard"], genre_dic["south carolina hip hop"], genre_dic["australian dance"], genre_dic["hollywood"], genre_dic["bass trap"], genre_dic["canadian rock"], genre_dic["san marcos tx indie"], genre_dic["british dance band"], genre_dic["vaudeville"], genre_dic["swing"], genre_dic["new jersey rap"], genre_dic["k-pop"], genre_dic["k-pop boy group"], genre_dic["indie pop rap"], genre_dic["brill building pop"],genre_dic["art pop"], genre_dic["popping"], genre_dic["canadian country"], genre_dic["queer country"], genre_dic["modern country pop"], genre_dic["atl trap"], genre_dic["library music"], genre_dic["laboratorio"], genre_dic["exotica"], genre_dic["memphis hip hop"], genre_dic["tennessee hip hop"], genre_dic["british indie rock"], genre_dic["indie rock"], genre_dic["chamber pop"], genre_dic["seattle indie"], genre_dic["stomp and holler"], genre_dic["viral pop"], genre_dic["light music"], genre_dic["electropowerpop"], genre_dic["philly rap"], genre_dic["ccm"], genre_dic["christian pop"], genre_dic["pop worship"], genre_dic["rap kreyol"], genre_dic["canadian rockabilly"], genre_dic["traditional rockabilly"], genre_dic["cowboy western"], genre_dic["western swing"], genre_dic["pluggnb"], genre_dic["new orleans blues"], genre_dic["piano blues"], genre_dic["louisiana blues"], genre_dic["baroque pop"], genre_dic["funk metal"], genre_dic["honky-tonk piano"], genre_dic["old west"], genre_dic["soul blues"], genre_dic["electro"], genre_dic["filter house"],
                  genre_dic["drill"], genre_dic["chicago drill"], genre_dic["arkansas country"], genre_dic["traditional country"], genre_dic["honky tonk"], genre_dic["electro house"], genre_dic["progressive house"], genre_dic["uk dance"], genre_dic["house"], genre_dic["edm"], genre_dic["new rave"], genre_dic["neo-synthpop"], genre_dic["escape room"], genre_dic["baton rouge rap"], genre_dic["shoegaze"], genre_dic["britpop"], genre_dic["melancholia"], genre_dic["tropical"], genre_dic["bounce"], genre_dic["classic uk pop"], genre_dic["reggaeton"], genre_dic["urbano latino"], genre_dic["trap latino"],genre_dic['"mans orchestra"'],genre_dic["brooklyn drill"], genre_dic["rock drums"], genre_dic["sleaze rock"], genre_dic["chicago bop"], genre_dic["dancehall"], genre_dic["trap queen"], genre_dic["country pop"], genre_dic["alternative rock"], genre_dic["pop r&b"], genre_dic["gambian hip hop"], genre_dic["new york drill"], genre_dic["uk drill"], genre_dic["melodic drill"], genre_dic["uk hip hop"], genre_dic["jazz blues"], genre_dic["christian alternative rock"], genre_dic["worship"], genre_dic["christian music"], genre_dic["rhythm and blues"], genre_dic['"childrens music"'], genre_dic["classic texas country"], genre_dic["reggae fusion"], genre_dic["sacramento indie"], genre_dic["rage rap"], genre_dic["glitchcore"], genre_dic["plugg"],genre_dic["outlaw country"], genre_dic["colombian pop"], genre_dic["latin pop"], genre_dic["big room"], genre_dic["pop dance"], genre_dic["australian rock"], genre_dic["jazz funk"], genre_dic["freakbeat"], genre_dic["ohio hip hop"], genre_dic["jangle pop"], genre_dic["emo rap"], genre_dic["chicago soul"], genre_dic["american orchestra"], genre_dic["orchestra"], genre_dic["classical"], genre_dic["swamp pop"], genre_dic["uk contemporary r&b"], genre_dic["lovers rock"], genre_dic["reggae"], genre_dic["vintage italian soundtrack"], genre_dic["italian library music"], genre_dic["tribal house"], genre_dic["brostep"], genre_dic["progressive electro house"], genre_dic["dfw rap"], genre_dic["punk"], genre_dic["big band"], genre_dic["dixieland"], genre_dic["jazz trombone"], genre_dic["athens indie"], genre_dic["uk reggae"], genre_dic["modern folk rock"],genre_dic["uk americana"], genre_dic["electronic trap"], genre_dic["gauze pop"], genre_dic["acoustic pop"], genre_dic["idol"], genre_dic["reggae maghreb"], genre_dic["rai"], genre_dic["downtempo"], genre_dic["australian electropop"], genre_dic["australian indie"], genre_dic["canadian old school hip hop"], genre_dic["classic canadian rock"], genre_dic["jazz"], genre_dic["bebop"], genre_dic["jazz saxophone"], genre_dic["lgbtq+ hip hop"], genre_dic["indie soul"], genre_dic["proto-metal"], genre_dic["acid rock"], genre_dic["alternative r&b"], genre_dic["political hip hop"], genre_dic["afrofuturism"], genre_dic["alternative hip hop"], genre_dic["soundtrack"], genre_dic["orchestral soundtrack"], genre_dic["australian pop"], genre_dic["jamaican hip hop"], genre_dic["post-punk"], genre_dic["gothic rock"], genre_dic["cancion melodica"], genre_dic["ranchera"], genre_dic["mariachi"], genre_dic["musica mexicana"], genre_dic["garage rock"], genre_dic["tempe indie"], genre_dic["oakland hip hop"], genre_dic["houston rap"], genre_dic["british soul"], genre_dic["mexican classic rock"], genre_dic["rap metal"], genre_dic["straight-ahead jazz"], genre_dic["comic"], genre_dic["drama"], genre_dic["slap house"], genre_dic["uk alternative pop"], genre_dic["electronica"], genre_dic["spanish invasion"], genre_dic["underground hip hop"],genre_dic["portland hip hop"], genre_dic["brit funk"], genre_dic["michigan indie"], genre_dic["swedish pop"], genre_dic["dmv rap"], genre_dic["motivation"], genre_dic["florida rap"], genre_dic["florida drill"], genre_dic["grunge"], genre_dic["spacegrunge"], genre_dic["mexican pop"], genre_dic["puerto rican pop"], genre_dic["reggaeton colombiano"], genre_dic["tin pan alley"], genre_dic["emo"], genre_dic["soul jazz"], genre_dic["north carolina hip hop"], genre_dic["complextro"], genre_dic["canadian trap"], genre_dic["st louis rap"], genre_dic["minnesota hip hop"], genre_dic["canadian psychedelic"], genre_dic["psychedelic folk"], genre_dic["beach music"], genre_dic["cali rap"], genre_dic["shiver pop"], genre_dic["jazz guitar"], genre_dic["anarcho-punk"], genre_dic["la indie"], genre_dic["modern alternative rock"], genre_dic["stomp pop"], genre_dic["golden age hip hop"], genre_dic["old school hip hop"], genre_dic["red dirt"], genre_dic["alternative country"], genre_dic["bluegrass"], genre_dic["new americana"], genre_dic["black americana"], genre_dic["progressive bluegrass"], genre_dic["trap soul"], genre_dic["pittsburgh rap"], genre_dic["harlem hip hop"], genre_dic["deep ragga"], genre_dic["old school dancehall"], genre_dic["folk-pop"], genre_dic["mambo"], genre_dic["dancehall queen"],genre_dic["reggaeton flow"], genre_dic["rap rock"], genre_dic["bass house"], genre_dic["tech house"], genre_dic["vintage jazz"], genre_dic["jazz clarinet"], genre_dic["roots reggae"], genre_dic["new orleans funk"], genre_dic["nigerian hip hop"], genre_dic["alte"], genre_dic["azonto"], genre_dic["nigerian pop"], genre_dic["afrobeats"], genre_dic["azontobeats"], genre_dic["afro r&b"], genre_dic["afropop"], genre_dic["classical tenor"], genre_dic["italian tenor"], genre_dic["chopped and screwed"], genre_dic["ballroom"], genre_dic["australian hip hop"], genre_dic["atlanta bass"], genre_dic["instrumental surf"], genre_dic["british folk"], genre_dic["nwobhm"], genre_dic["lds"], genre_dic["classical performance"],genre_dic["american choir"], genre_dic["orchestral performance"], genre_dic["cubaton"], genre_dic["electro latino"], genre_dic["etherpop"], genre_dic["indie poptimism"], genre_dic["futuristic swag"], genre_dic["rock of gibraltar"], genre_dic["anti-folk"], genre_dic["roots rock"], genre_dic["underground power pop"], genre_dic["swedish synthpop"], genre_dic["swedish electropop"], genre_dic["scandipop"], genre_dic["modern reggae"], genre_dic["mississippi hip hop"], genre_dic["deep disco"], genre_dic["neo r&b"], genre_dic["ghanaian hip hop"], genre_dic["dansktop"], genre_dic["gospel"], genre_dic["gospel r&b"], genre_dic["modern blues"], genre_dic["canadian blues"], genre_dic["canzone napoletana"], genre_dic["jesus movement"], genre_dic["swedish melodic rock"], genre_dic["swedish hard rock"], genre_dic["jazz fusion"], genre_dic["theme"], genre_dic["zolo"], genre_dic["memphis blues"], genre_dic["chicano punk"], genre_dic["german pop"], genre_dic["neue deutsche welle"], genre_dic["asian american hip hop"], genre_dic["pinoy hip hop"], genre_dic["early synthpop"], genre_dic["moog"], genre_dic["traditional blues"],genre_dic["acoustic blues"], genre_dic["swamp blues"], genre_dic["harmonica blues"], genre_dic["melbourne bounce international"], genre_dic["dutch rock"], genre_dic["dutch pop"], genre_dic["hammond organ"], genre_dic["cantautor"], genre_dic["latin arena pop"], genre_dic["paisley underground"], genre_dic["moombahton"], genre_dic["danish pop"], genre_dic["virginia hip hop"], genre_dic["miami bass"], genre_dic["halloween"], genre_dic["native american"], genre_dic["jazz organ"], genre_dic["canadian latin"], genre_dic["song poem"], genre_dic["scottish singer-songwriter"], genre_dic["new mexico music"], genre_dic["chattanooga indie"], genre_dic["supergroup"], genre_dic["dutch house"], genre_dic["rap conscient"], genre_dic["vapor trap"], genre_dic["alaska indie"], genre_dic["ska revival"], genre_dic["ska"], genre_dic["alternative pop"], genre_dic["kentucky hip hop"], genre_dic["deep underground hip hop"], genre_dic["vintage schlager"], genre_dic["classic schlager"], genre_dic["yu-mex"], genre_dic["croatian pop"], genre_dic["irish singer-songwriter"], genre_dic["musica para ninos"], genre_dic["country gospel"], genre_dic["bluegrass gospel"], genre_dic["country boogie"],
                  genre_dic["yodeling"], genre_dic["cloud rap"], genre_dic["art punk"], genre_dic["new jersey underground rap"], genre_dic["nz pop"], genre_dic["socal pop punk"], genre_dic["classic swedish pop"], genre_dic["deep freestyle"], genre_dic["bouncy house"], genre_dic["jam band"], genre_dic["hyphy"], genre_dic["madchester"], genre_dic["noise pop"], genre_dic["comedy rap"], genre_dic["zither"], genre_dic["german soundtrack"], genre_dic["synthesizer"], genre_dic["souldies"], genre_dic["comic metal"], genre_dic["country rap"], genre_dic["soca"], genre_dic["vincy soca"], genre_dic["samba-jazz"], genre_dic["violao"], genre_dic["bossa nova"], genre_dic["latin jazz"], genre_dic["brazilian jazz"], genre_dic["cool jazz"], genre_dic["christian lo-fi"], genre_dic["samba"], genre_dic["francoton"], genre_dic["bossbeat"], genre_dic["rebel blues"], genre_dic["battle rap"], genre_dic["banjo"], genre_dic["acid jazz"], genre_dic["jazz rap"], genre_dic["boston hip hop"], genre_dic["middle earth"], genre_dic["celtic"], genre_dic["pop edm"], genre_dic["sped up"], genre_dic["australian country"], genre_dic["fake"], genre_dic["irish pub song"], genre_dic["irish folk"], genre_dic["canadian celtic"], genre_dic["nueva ola chilena"], genre_dic["native american contemporary"],genre_dic["new beat"], genre_dic["scottish rock"], genre_dic["k-rap"], genre_dic["korean old school hip hop"], genre_dic["new orleans jazz"], genre_dic["cyberpunk"], genre_dic["cosmic american"], genre_dic["psychedelic folk rock"], genre_dic["screamo"], genre_dic["post-disco soul"], genre_dic["hyperpop"], genre_dic["comedy"], genre_dic["broadway"], genre_dic["show tunes"], genre_dic["indie r&b"], genre_dic["bedroom r&b"], genre_dic["british orchestra"], genre_dic["comedy rock"], genre_dic["italian metal"], genre_dic["grebo"], genre_dic["rhode island rap"], genre_dic["classic house"], genre_dic["cologne indie"], genre_dic["german singer-songwriter"], genre_dic["rap latina"], genre_dic["latin viral pop"], genre_dic["dong-yo"], genre_dic["vintage hollywood"],genre_dic["latin funk"], genre_dic["speedrun"], genre_dic["video game music"], genre_dic["modern salsa"], genre_dic["salsa"], genre_dic["instrumental worship"], genre_dic["social media pop"], genre_dic["jazz quartet"], genre_dic["neo-singer-songwriter"], genre_dic["dutch edm"], genre_dic["deep soft rock"], genre_dic["jamaican ska"], genre_dic["bedroom pop"], genre_dic["modern alternative pop"], genre_dic["classic praise"], genre_dic["south african jazz"], genre_dic["derby indie"], genre_dic["celtic rock"], genre_dic["classic soundtrack"], genre_dic["jazz boom bap"], genre_dic["grime"], genre_dic["italo dance"], genre_dic["italian adult pop"], genre_dic["parody"], genre_dic['"childrens story"'], genre_dic["pub rock"], genre_dic["brooklyn indie"], genre_dic["uk funky"], genre_dic["modern uplift"], genre_dic["transpop"], genre_dic["modern blues rock"], genre_dic["rare groove"], genre_dic["french romanticism"], genre_dic["violin"], genre_dic["late romantic era"], genre_dic["la pop"], genre_dic["rochester ny indie"], genre_dic["electrofox"], genre_dic["nu disco"], genre_dic["bass music"], genre_dic["bergen indie"], genre_dic["west end"], genre_dic["italo house"], genre_dic["redneck"], genre_dic["indie hip hop"], genre_dic["truck-driving country"], genre_dic["jazz brass"], genre_dic["psychedelic hip hop"], genre_dic["texas blues"], genre_dic["harlem renaissance"], genre_dic["jamaican dancehall"], genre_dic["scottish new wave"], genre_dic["relaxative"], genre_dic["gregorian dance"], genre_dic["british soundtrack"], genre_dic["cowpunk"], genre_dic["vapor twitch"], genre_dic["norwegian pop"], genre_dic["detroit trap"], genre_dic["swedish country"],genre_dic["austropop"], genre_dic["slow game"], genre_dic["psychedelic blues-rock"], genre_dic["early reggae"], genre_dic["rocksteady"], genre_dic["experimental"], genre_dic["experimental pop"], genre_dic["south african rock"], genre_dic["jazz piano"], genre_dic["electric bass"], genre_dic["a cappella"], genre_dic["bossa nova jazz"], genre_dic["boogie"], genre_dic["romantico"], genre_dic["scam rap"], genre_dic["viral rap"], genre_dic["barbershop"], genre_dic["american oi"], genre_dic["jazz vibraphone"], genre_dic["indiecoustica"], genre_dic["jazz pop"], genre_dic["contemporary vocal jazz"], genre_dic["speed metal"], genre_dic["us power metal"], genre_dic["progressive metal"], genre_dic["indie pop"], genre_dic["contemporary post-bop"], genre_dic["hard bop"], genre_dic["enka"], genre_dic["modern power pop"], genre_dic["trance"], genre_dic["dream trance"], genre_dic["gen z singer-songwriter"], genre_dic["old school thrash"], genre_dic["thrash metal"], genre_dic["pop electronico"], genre_dic["cumbia"], genre_dic["latin rock"], genre_dic["latin alternative"], genre_dic["tropical alternativo"], genre_dic["industrial metal"], genre_dic["arkansas hip hop"], genre_dic["bedroom soul"], genre_dic["indie rock italiano"], genre_dic["italian pop"], genre_dic["aussietronica"], genre_dic["alternative dance"], genre_dic["taiwan electronic"], genre_dic["black comedy"], genre_dic["german rock"], genre_dic["german metal"], genre_dic["german hard rock"], genre_dic["german house"], genre_dic["deep dance pop"], genre_dic["nursery"], genre_dic["hel"], genre_dic["nashville indie"], genre_dic["alabama rap"], genre_dic["boston rock"], genre_dic["pop romantico"], genre_dic["romanian house"], genre_dic["moldovan pop"], genre_dic["romanian pop"],is_local
]])
    X_regression= X_regression.astype(float)
    print(X_regression.shape)
    Regression(X_regression)
    X_clas = np.array([[rank, safe_log(song_duration), math.sqrt(Acoustincess), Danceability, Energy,
                        safe_log(Instrumentalness), safe_log(Liveness), Loudness, safe_log(Speechness), Tempo, Valence,
                        Key, TimeSignature, safe_log(no_of_words), safe_log(no_of_chars), num_of_available_market,
                        year_rank, num_words_description, safe_log(num_entered_genres), safe_log(weight),
                        safe_log(total_weight), day, month, Year, int(is_catchy==0), int(is_catchy==1), int(int(is_colorful) == 0),
                        int(int(is_colorful) == 1), int(int(has_potential) == 0), int(int(has_potential) == 1),
                       int(Top_100_art_sc == 0), int(Top_100_art_sc == 1), int(mode == 0), int(mode == 1),
                       decade["Decade_Eighties"], decade["Decade_Fifties"], decade["Decade_Fourties"],
                       decade["Nighnties"], decade["Dcade_old"], decade["Decade_Seventies"], decade["Decade_Sixties"],
                       decade["Decade_Thirties"], decade["Decade_new millennium"], seasons["Autumn"], seasons["spring"],
                       seasons["Summer"], seasons["Winter"], int(duration_Category == "average_time"),
                       int(duration_Category == "small_time"), int(duration_Category == "large_time"), int(clust_no==0), int(clust_no==1), int(clust_no==2), int(clust_no==3), int(clust_no==4), int(clust_no==5),
                       int(clust_no==6), int(clust_no==7), int(clust_no==8), int(clust_no==9), int(clust_no==10), int(clust_no==11), int(clust_no==12),int(clust_no==13),int(clust_no==14),int(clust_no==15),int(clust_no==16),int(clust_no==17),int(clust_no==18),int(clust_no==19), int(Top_1800_Album == 0), int(Top_1800_Album == 1),genre_dic["pop"], genre_dic["dance pop"], genre_dic["rap metal"], genre_dic["post-grunge"], genre_dic["rock"], genre_dic["nu metal"], genre_dic["alternative metal"], genre_dic["dirty south rap"], genre_dic["gangster rap"], genre_dic["atl hip hop"], genre_dic["trap"], genre_dic["hip hop"], genre_dic["southern hip hop"], genre_dic["pop rap"], genre_dic["rap"], genre_dic["soul"], genre_dic["r&b"], genre_dic["novelty"], genre_dic["soul blues"], genre_dic["classic soul"], genre_dic["piano blues"], genre_dic["vocal jazz"], genre_dic["jazz blues"], genre_dic["miami hip hop"], genre_dic["adult standards"], genre_dic["rockabilly"], genre_dic["rock-and-roll"], genre_dic["easy listening"], genre_dic["hardcore hip hop"], genre_dic["hip pop"], genre_dic["philly rap"], genre_dic["battle rap"], genre_dic["urban contemporary"], genre_dic["detroit hip hop"], genre_dic["barbadian pop"], genre_dic["native american"], genre_dic["native american contemporary"], genre_dic["country road"], genre_dic["country"], genre_dic["country dawn"], genre_dic["country pop"], genre_dic["contemporary country"], genre_dic["uk pop"], genre_dic["mellow gold"], genre_dic["piano rock"], genre_dic["glam rock"], genre_dic["acid rock"], genre_dic["proto-metal"], genre_dic["protopunk"], genre_dic["british blues"], genre_dic["psychedelic rock"], genre_dic["british invasion"], genre_dic["hard rock"], genre_dic["album rock"], genre_dic["soft rock"], genre_dic["yacht rock"], genre_dic["classic rock"], genre_dic["singer-songwriter"], genre_dic["folk"], genre_dic["folk rock"], genre_dic["bubblegum pop"], genre_dic["lounge"], genre_dic["surf music"], genre_dic["alt z"], genre_dic["transpop"], genre_dic["stomp pop"], genre_dic["modern alternative rock"], genre_dic["modern rock"], genre_dic["deep adult standards"], genre_dic["norwegian pop"], genre_dic["vapor twitch"], genre_dic["funk"], genre_dic["quiet storm"], genre_dic["canadian pop"], genre_dic["canadian hip hop"], genre_dic["east coast hip hop"], genre_dic["west coast rap"], genre_dic["conscious hip hop"],genre_dic["toronto rap"], genre_dic["karaoke"], genre_dic["canadian contemporary r&b"], genre_dic["electro"], genre_dic["filter house"], genre_dic["atl trap"], genre_dic["modern country rock"], genre_dic["miami bass"], genre_dic["atlanta bass"], genre_dic["minneapolis sound"], genre_dic["funk rock"], genre_dic["synth funk"], genre_dic["crunk"], genre_dic["north carolina hip hop"], genre_dic["virginia hip hop"], genre_dic["neo soul"], genre_dic["queens hip hop"], genre_dic["heartland rock"], genre_dic["girl group"], genre_dic["rap kreyol"], genre_dic["vaudeville"], genre_dic["swing"], genre_dic["british dance band"], genre_dic["irish folk"], genre_dic["canadian celtic"], genre_dic["irish pub song"], genre_dic["talent show"], genre_dic["rap rock"], genre_dic["lilith"], genre_dic["rhythm and blues"], genre_dic["southern soul"], genre_dic["chicago rap"], genre_dic["contemporary r&b"], genre_dic["boy band"], genre_dic["new jack swing"], genre_dic["lgbtq+ hip hop"], genre_dic["motown"], genre_dic["disco"], genre_dic["new wave pop"], genre_dic["beatlesque"], genre_dic["rock drums"], genre_dic["candy pop"], genre_dic["jangle pop"], genre_dic["memphis soul"], genre_dic["philly soul"], genre_dic["permanent wave"], genre_dic["beach music"], genre_dic["instrumental soul"], genre_dic["new wave"], genre_dic["glam metal"], genre_dic["new romantic"], genre_dic["country rock"], genre_dic["instrumental funk"], genre_dic["blues"], genre_dic["synthpop"], genre_dic["nashville sound"], genre_dic["progressive rock"], genre_dic["electro house"], genre_dic["house"], genre_dic["edm"], genre_dic["progressive house"], genre_dic["uk dance"], genre_dic["pop rock"], genre_dic["neo mellow"], genre_dic["sophisti-pop"], genre_dic["power pop"], genre_dic["doo-wop"], genre_dic["british folk"], genre_dic["psychedelic folk"], genre_dic["scottish singer-songwriter"], genre_dic["classic country pop"], genre_dic["blues rock"], genre_dic["halloween"], genre_dic["art rock"], genre_dic["pop punk"], genre_dic["neon pop punk"], genre_dic["classic uk pop"], genre_dic["trap queen"], genre_dic["symphonic rock"], genre_dic["detroit rock"], genre_dic["metal"], genre_dic["fake"], genre_dic["memphis hip hop"], genre_dic["pittsburgh rap"], genre_dic["electropop"],
                       genre_dic["melodic rap"], genre_dic["metropopolis"], genre_dic["new jersey rap"], genre_dic["alternative r&b"], genre_dic["g funk"], genre_dic["old school atlanta hip hop"], genre_dic["new orleans rap"], genre_dic["brill building pop"], genre_dic["roots rock"], genre_dic["classic oklahoma country"], genre_dic["rage rap"], genre_dic["british orchestra"], genre_dic["sunshine pop"], genre_dic["orchestra"], genre_dic["baroque pop"], genre_dic["torch song"], genre_dic["space age pop"], genre_dic["merseybeat"], genre_dic["post-teen pop"], genre_dic["german hard rock"], genre_dic["german rock"], genre_dic["german metal"], genre_dic["europop"], genre_dic["eurodance"], genre_dic["viral trap"], genre_dic["neo r&b"], genre_dic["ghanaian hip hop"], genre_dic["jazz trumpet"], genre_dic["german pop"], genre_dic["neue deutsche welle"], genre_dic["australian dance"], genre_dic["australian pop"], genre_dic["chicano punk"], genre_dic["classic garage rock"], genre_dic["sleaze rock"], genre_dic["psychedelic soul"], genre_dic["p funk"], genre_dic["pop dance"], genre_dic["pop edm"], genre_dic["progressive electro house"], genre_dic["ohio hip hop"], genre_dic["pop soul"], genre_dic["british soul"], genre_dic["spanish invasion"], genre_dic["afrofuturism"], genre_dic["canadian rock"], genre_dic["sacramento indie"], genre_dic["exotica"], genre_dic["library music"], genre_dic["laboratorio"], genre_dic["post-disco"], genre_dic["freestyle"], genre_dic["jazz funk"], genre_dic["funk metal"], genre_dic["pop r&b"], genre_dic["black americana"], genre_dic["grime"], genre_dic["uk contemporary r&b"], genre_dic["cali rap"], genre_dic["oakland hip hop"], genre_dic["hyphy"], genre_dic["uk alternative pop"], genre_dic["pov: indie"], genre_dic["mambo"], genre_dic["harlem hip hop"], genre_dic["southern rock"], genre_dic["dancehall"], genre_dic["mexican classic rock"], genre_dic["viral pop"], genre_dic["canadian singer-songwriter"], genre_dic["canadian country"], genre_dic["hip house"], genre_dic["south carolina hip hop"], genre_dic["dance rock"], genre_dic["art pop"], genre_dic["bubblegum dance"], genre_dic["vintage hollywood"], genre_dic["electric blues"], genre_dic["sped up"], genre_dic["hyperpop"], genre_dic["american folk revival"], genre_dic["reggae fusion"], genre_dic["deep talent show"], genre_dic["nederpop"], genre_dic["dutch prog"], genre_dic["dutch rock"], genre_dic["vocal harmony group"], genre_dic["classic girl group"], genre_dic["chicago bop"], genre_dic["rare groove"], genre_dic["diva house"], genre_dic["indietronica"], genre_dic["rhode island rap"], genre_dic["emo rap"], genre_dic["houston rap"], genre_dic["punk"], genre_dic["texas blues"], genre_dic["modern blues"], genre_dic["grunge"], genre_dic["alternative rock"], genre_dic["hi-nrg"], genre_dic["gospel"], genre_dic["gospel r&b"],
                       genre_dic["nyc rap"], genre_dic["chicago soul"], genre_dic["souldies"], genre_dic["mexican pop"], genre_dic["latin pop"], genre_dic["indie pop rap"], genre_dic["post-punk"], genre_dic["zolo"], genre_dic["northern soul"], genre_dic["irish singer-songwriter"], genre_dic["underground power pop"], genre_dic["louisiana blues"], genre_dic["new orleans blues"], genre_dic["swing italiano"], genre_dic["golden age hip hop"], genre_dic["old school hip hop"], genre_dic["classic canadian rock"], genre_dic["german techno"], genre_dic["cowboy western"], genre_dic["garage rock"], genre_dic["florida rap"], genre_dic["trap latino"], genre_dic["florida drill"], genre_dic["reggaeton colombiano"], genre_dic["urbano latino"], genre_dic["reggaeton"], genre_dic["michigan indie"], genre_dic["canadian psychedelic"], genre_dic["theme"], genre_dic["idol"], genre_dic["ectofolk"], genre_dic["etherpop"], genre_dic["indie poptimism"], genre_dic["tropical house"], genre_dic["classic praise"], genre_dic["christian music"], genre_dic["comic"], genre_dic["australian rock"], genre_dic["movie tunes"], genre_dic["irish rock"], genre_dic["arkansas country"], genre_dic["francoton"], genre_dic["comic metal"], genre_dic['"mans orchestra"'], genre_dic["rebel blues"], genre_dic["bossbeat"], genre_dic["big band"], genre_dic["new beat"], genre_dic["st louis rap"], genre_dic["chopped and screwed"], genre_dic["vocal house"], genre_dic["honky-tonk piano"], genre_dic["traditional blues"], genre_dic["acoustic blues"], genre_dic["shiver pop"], genre_dic["gauze pop"], genre_dic["puerto rican pop"], genre_dic["outlaw country"], genre_dic["swedish pop"], genre_dic["gambian hip hop"], genre_dic["brooklyn drill"], genre_dic["chicago drill"], genre_dic["melodic drill"], genre_dic["drill"], genre_dic["uk hip hop"], genre_dic["uk drill"], genre_dic["new york drill"], genre_dic["smooth jazz"], genre_dic["acoustic pop"], genre_dic["rap conscient"], genre_dic["vapor trap"], genre_dic["colombian pop"], genre_dic["roots reggae"], genre_dic["reggae"], genre_dic["bounce"], genre_dic["tempe indie"], genre_dic["k-pop boy group"], genre_dic["k-pop"], genre_dic["bass trap"], genre_dic["dfw rap"], genre_dic["swedish synthpop"], genre_dic["swedish electropop"], genre_dic["straight-ahead jazz"], genre_dic["jazz brass"], genre_dic["soul jazz"], genre_dic["jazz guitar"], genre_dic["jazz"], genre_dic["bebop"], genre_dic["jazz trombone"], genre_dic["bedroom pop"], genre_dic["bedroom r&b"], genre_dic["jamaican ska"], genre_dic["ballroom"], genre_dic["operatic pop"], genre_dic["chicano rap"], genre_dic["texas latin rap"], genre_dic["latin hip hop"], genre_dic["truck-driving country"], genre_dic["brostep"], genre_dic["electronic trap"], genre_dic['"childrens music"'], genre_dic["a cappella"], genre_dic["melancholia"], genre_dic["drama"], genre_dic["modern country pop"], genre_dic["tin pan alley"], genre_dic["dixieland"], genre_dic["vintage jazz"], genre_dic["modern blues rock"], genre_dic["azontobeats"], genre_dic["afropop"], genre_dic["azonto"], genre_dic["afro r&b"], genre_dic["alte"], genre_dic["nigerian hip hop"], genre_dic["nigerian pop"], genre_dic["afrobeats"], genre_dic["big room"], genre_dic["new mexico music"], genre_dic["tropical"], genre_dic["modern salsa"], genre_dic["salsa"], genre_dic["melbourne bounce international"], genre_dic["smooth saxophone"], genre_dic["canadian blues"], genre_dic["brit funk"], genre_dic["zither"], genre_dic["ska"], genre_dic["britpop"], genre_dic["ska revival"], genre_dic["uk funky"], genre_dic["rock keyboard"], genre_dic["indie soul"], genre_dic["trip hop"], genre_dic["nwobhm"], genre_dic["tennessee hip hop"], genre_dic["dutch pop"], genre_dic["swamp pop"], genre_dic["broadway"], genre_dic["classic soundtrack"], genre_dic["popping"], genre_dic["barbershop"], genre_dic["french romanticism"], genre_dic["violin"], genre_dic["late romantic era"], genre_dic["bronx hip hop"], genre_dic["instrumental surf"], genre_dic["music hall"], genre_dic["kentucky hip hop"], genre_dic["australian electropop"], genre_dic["alternative dance"], genre_dic["aussietronica"],
                       genre_dic["neo-synthpop"], genre_dic["escape room"], genre_dic["new rave"], genre_dic["complextro"], genre_dic["western swing"], genre_dic["australian country"], genre_dic["mississippi hip hop"], genre_dic["scandipop"], genre_dic["glam punk"], genre_dic["flute rock"], genre_dic["hammond organ"], genre_dic["slap house"], genre_dic["korean old school hip hop"], genre_dic["k-rap"], genre_dic["experimental"], genre_dic["experimental pop"], genre_dic["orchestral soundtrack"], genre_dic["soundtrack"], genre_dic["austropop"], genre_dic["baton rouge rap"], genre_dic["underground hip hop"], genre_dic["speedrun"], genre_dic["video game music"], genre_dic["deep underground hip hop"], genre_dic["stomp and holler"], genre_dic["uk americana"], genre_dic["modern folk rock"], genre_dic["reggaeton flow"], genre_dic["country gospel"], genre_dic["bluegrass gospel"], genre_dic["country boogie"], genre_dic["yodeling"], genre_dic["scam rap"], genre_dic["classic texas country"], genre_dic["screamo"], genre_dic["athens indie"], genre_dic["lovers rock"], genre_dic["uk reggae"], genre_dic["australian hip hop"], genre_dic["new orleans jazz"], genre_dic["supergroup"], genre_dic["hollywood"], genre_dic["paisley underground"], genre_dic["folk-pop"], genre_dic["moombahton"], genre_dic["jazz organ"], genre_dic["canzone napoletana"], genre_dic["classical tenor"], genre_dic["latin arena pop"], genre_dic["cantautor"], genre_dic["grebo"], genre_dic["madchester"], genre_dic["new orleans funk"], genre_dic["new orleans soul"], genre_dic["plugg"], genre_dic["pluggnb"], genre_dic["swamp rock"], genre_dic["german house"], genre_dic["vintage italian soundtrack"], genre_dic["italian library music"], genre_dic["jesus movement"], genre_dic["taiwan electronic"], genre_dic["electro latino"], genre_dic["cubaton"], genre_dic["jamaican hip hop"], genre_dic["neo-singer-songwriter"], genre_dic["south african rock"], genre_dic["alabama rap"], genre_dic["enka"], genre_dic["new americana"], genre_dic["italo house"], genre_dic["political hip hop"], genre_dic["cloud rap"], genre_dic["dutch house"], genre_dic["deep dance pop"], genre_dic["new jersey underground rap"], genre_dic["dmv rap"], genre_dic["trap soul"], genre_dic["jazz saxophone"], genre_dic["spacegrunge"], genre_dic["emo"], genre_dic["jazz boom bap"], genre_dic["samba-jazz"], genre_dic["violao"], genre_dic["bossa nova"], genre_dic["latin jazz"], genre_dic["christian lo-fi"], genre_dic["samba"], genre_dic["cool jazz"], genre_dic["brazilian jazz"], genre_dic["jazz rap"], genre_dic["psychedelic hip hop"], genre_dic["industrial metal"], genre_dic["early synthpop"], genre_dic["moog"], genre_dic["alternative hip hop"], genre_dic["minnesota hip hop"], genre_dic["rochester ny indie"], genre_dic["la pop"], genre_dic["big beat"], genre_dic["ambient house"], genre_dic["acid house"], genre_dic["american oi"], genre_dic["soca"], genre_dic["dancehall queen"],
                       genre_dic["jamaican dancehall"], genre_dic["latin viral pop"], genre_dic["rap latina"], genre_dic["pixie"], genre_dic["indie rock"], genre_dic["canadian trap"], genre_dic["musica para ninos"], genre_dic["canadian latin"], genre_dic["nashville indie"], genre_dic["west end"], genre_dic["brooklyn indie"], genre_dic["futuristic swag"], genre_dic["nz pop"], genre_dic["south african jazz"], genre_dic["nursery"], genre_dic["deep soft rock"], genre_dic["pop romantico"], genre_dic["cancion melodica"], genre_dic["ranchera"], genre_dic["boston hip hop"], genre_dic["electropowerpop"], genre_dic["danish pop"], genre_dic["arkansas hip hop"], genre_dic["jam band"], genre_dic["electrofox"], genre_dic["nu disco"], genre_dic["deep disco"], genre_dic["old school thrash"], genre_dic["thrash metal"], genre_dic["country rap"], genre_dic["old west"], genre_dic["socal pop punk"], genre_dic["bergen indie"], genre_dic['"womens music"'], genre_dic["gen z singer-songwriter"], genre_dic["deep freestyle"], genre_dic["traditional country"], genre_dic["celtic rock"], genre_dic["seattle hip hop"], genre_dic["freakbeat"], genre_dic["pop worship"], genre_dic["ccm"], genre_dic["christian pop"], genre_dic["jazz rock"], genre_dic["san marcos tx indie"], genre_dic["italian disco"], genre_dic["cowpunk"], genre_dic["persian pop"], genre_dic["comedy rock"], genre_dic["indie rock italiano"], genre_dic["italian pop"], genre_dic["anarcho-punk"], genre_dic["slow game"], genre_dic["latin funk"], genre_dic["pop emo"], genre_dic["glitchcore"], genre_dic["christian alternative rock"], genre_dic["worship"], genre_dic["comedy"], genre_dic["black comedy"], genre_dic["motivation"], genre_dic["gothic rock"], genre_dic["light music"], genre_dic["italian tenor"], genre_dic["reggae maghreb"], genre_dic["rai"], genre_dic["uk post-punk"], genre_dic["dong-yo"], genre_dic["classic swedish pop"], genre_dic["progressive bluegrass"], genre_dic["bluegrass"], genre_dic["alternative country"], genre_dic["red dirt"], genre_dic["indie pop"], genre_dic["classical"], genre_dic["british soundtrack"], genre_dic["bass music"], genre_dic["scottish rock"], genre_dic["swedish country"], genre_dic["comedy rap"], genre_dic["psychedelic folk rock"], genre_dic["redneck"], genre_dic["traditional rockabilly"], genre_dic["scottish new wave"], genre_dic["swamp blues"], genre_dic["harmonica blues"], genre_dic["italian metal"], genre_dic["dream trance"], genre_dic["trance"], genre_dic["canadian old school hip hop"], genre_dic["rock of gibraltar"], genre_dic["mariachi"], genre_dic["musica mexicana"], genre_dic["jazz clarinet"], genre_dic["song poem"], genre_dic["synthesizer"], genre_dic["cyberpunk"], genre_dic["honky tonk"], genre_dic["social media pop"], genre_dic["pinoy hip hop"], genre_dic["birmingham metal"], genre_dic["hel"], genre_dic["dutch edm"], genre_dic["relaxative"], genre_dic["american orchestra"], genre_dic["electronica"], genre_dic["contemporary vocal jazz"], genre_dic["jazz pop"], genre_dic["indie r&b"], genre_dic["tribal house"], genre_dic["progressive metal"], genre_dic["us power metal"], genre_dic["speed metal"], genre_dic["modern power pop"], genre_dic["romantico"], genre_dic["asian american hip hop"], genre_dic["moldovan pop"], genre_dic["romanian pop"], genre_dic["romanian house"], genre_dic["jazz fusion"], genre_dic["shoegaze"], genre_dic["indiecoustica"], genre_dic["viral rap"], genre_dic["alaska indie"], genre_dic["classic house"], genre_dic["otacore"], genre_dic["memphis blues"], genre_dic["instrumental worship"], genre_dic["downtempo"], genre_dic["australian indie"], genre_dic["boston rock"], genre_dic["bouncy house"], genre_dic["bossa nova jazz"], genre_dic["italian adult pop"], genre_dic["italo dance"], genre_dic["la indie"], genre_dic["wrestling"], genre_dic["jazz vibraphone"], genre_dic["gregorian dance"], genre_dic["boogie"], genre_dic["hard bop"], genre_dic["contemporary post-bop"], genre_dic["celtic"], genre_dic["middle earth"], genre_dic["military cadence"], genre_dic["cosmic american"]

]])
    Classification(X_clas)
    print(X_clas.shape)
def Regression(X):
   with open('XGB_Regression.pkl', 'rb') as f:
      model = pickle.load(f)
   
   predictions = model.predict(X)[0]
   predictions = int(predictions)
   if (predictions<0):
      predictions=0
   if(predictions>100):
      predictions = 100

   print(predictions)
   st.write("Popularity : ",predictions)
def Classification(X):
    scaler = pickle.load(open(r"scalerMinMax.pkl","rb"))
    scaled_x = scaler.transform(X)
    New_X = np.zeros((1, 370))
    index = [  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
         13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,
         26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  39,
         40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,
         54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  65,  66,  67,
         68,  69,  70,  71,  72,  73,  75,  76,  77,  78,  79,  80,  81,
         82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,
         95,  96,  97,  98,  99, 100, 101, 102, 104, 105, 106, 107, 109,
        110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122,
        123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135,
        138, 139, 140, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152,
        156, 157, 158, 159, 160, 161, 162, 164, 165, 166, 167, 168, 170,
        171, 172, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186,
        187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199,
        200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212,
        213, 214, 215, 216, 217, 220, 221, 223, 224, 225, 226, 227, 228,
        229, 230, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242,
        243, 244, 247, 249, 250, 251, 252, 253, 257, 258, 259, 262, 263,
        264, 265, 266, 268, 269, 270, 271, 272, 274, 275, 276, 277, 280,
        282, 283, 285, 286, 287, 288, 290, 292, 293, 294, 297, 299, 300,
        301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 312, 313, 315,
        316, 317, 318, 321, 322, 323, 324, 325, 326, 328, 333, 334, 335,
        338, 339, 341, 342, 346, 349, 352, 353, 354, 355, 356, 358, 359,
        360, 361, 362, 363, 366, 367, 368, 371, 373, 374, 375, 376, 377,
        378, 384, 386, 389, 394, 395, 396, 398, 405, 406, 410, 411, 422,
        423, 424, 428, 431, 432, 434, 436, 437, 440, 442, 443, 444, 446,
        454, 455, 466, 469, 470, 473, 483, 486, 491, 493, 494, 500, 501,
        510, 523, 527, 528, 531, 543, 546, 558, 559, 563, 598, 602, 617,
        618, 632, 640, 663, 682, 685]
    for i in range(370):
       New_X[:,i] = scaled_x[:,index[i]]
    with open("model.pkl", 'rb') as f:
        model = pickle.load(f)
    y = model.predict(New_X)[0]
    
    with open("LableEncoder.pkl", 'rb') as f:
     label_encoder = pickle.load(f)
    encoded_labels =  label_encoder.inverse_transform([y])

    st.write("your prediction : ",encoded_labels)
    print(y)

    
def main():
    st.title('Spotify Song Popularity Prediction')
    st.image("""https://storage.googleapis.com/pr-newsroom-wp/1/2018/11/Spotify_Logo_RGB_Green.png""")
    #Song_Name
    Song_Name = st.text_input("Enter Song Name :")

    #Album_Name
    Album_Name = st.text_input("Enter Album Name :")


    # genre_dic input field
    genre_dics = st.text_input("Enter genre_dics (separated by Space):")

    #Artist_Name
    Artist_Name = st.text_input("Enter Artist Name :")

    #Acousticness
    Acousticness = st.number_input("acousticness:",  min_value=0.0, max_value=0.99,step=0.1)

    #Loudness
    Loudness = st.number_input('Loudness:', min_value=-37.841, max_value= -0.81,step=0.1)

    #Instrumentals
    Instrumentals = st.number_input('Instrumentals:', min_value=0.0, max_value= 0.97,step=0.1)

    #Liveness
    Liveness = st.number_input('Loudness:', min_value=0.0, max_value= 0.9,step=0.1)

    #Danceability
    Danceability = st.number_input('Danceability:', min_value=0.0, max_value= 0.9,step=0.1)

    #Valence
    Valence = st.number_input('Rank:', min_value=0.0, max_value= 0.99,step=0.1)

    
    
    # Mode selection
    mode = st.selectbox("Select Mode:", [0, 1])
    #Date
    Date = st.text_input('Date:')


    #Energy
    Energy = st.number_input('Energy:', min_value=0.0, max_value= 0.9,step=0.1)

    #Tempo
    Tempo = st.number_input('Tempo:', min_value=0.0, max_value= 232.0,step=1.1)

    # Time Signature
    TimeSignature = st.number_input('Time Signature:', min_value=0.0, max_value=5.0, step=1.0)

    # Key
    Key = st.number_input('Key:', min_value=0.0, max_value= 11.0, step=1.0)

    

    # Speechness
    Speechness = st.number_input('Speechness:', min_value=0.0, max_value=0.9, step=0.0001)

    
    #song_duration
    
    song_duration = st.number_input('Enter song duration(ms):', min_value=0, step=1)
    #Song_Img_Url
   
    Song_img_URL = st.text_input("Song Image URL :") #PASS AS PARAM IN PREDICT FUNCTION 
    if st.button('Predict '):
      predict(genre_dics,Song_Name,Artist_Name,Album_Name,Date,song_duration,Acousticness,Danceability,Energy,Instrumentals,Liveness,Loudness,Speechness,Tempo,Valence,Key,TimeSignature,mode,Song_img_URL)
    
    
    

# Run the app
if __name__ == "__main__":
    main()
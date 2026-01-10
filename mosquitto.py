# ***************************************************************************************************************
# 											import
# ***************************************************************************************************************
#!python3
import time
import sys
import pandas as pd
#mqtt
import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish


# ***************************************************************************************************************
# 											import donnée 
# ***************************************************************************************************************
#lecture des csv
df=pd.read_csv("result.csv",usecols=[1])	#on prend la colonne des donnees et on la met dans une dataframe de pandas
df1=pd.read_csv("result1.csv",usecols=[1])
df2=pd.read_csv("result2.csv",usecols=[1])

#creation des labels des labels
label1 = [1 for i in range(len(df))] 	#cree une liste de 1 de meme longueure que la dataframe
label2 = [2 for i in range(len(df1))]
label3 = [3 for i in range(len(df2))]
#convertion des labels en Series pandas
value_label1 = pd.Series(label1)
value_label2 = pd.Series(label2)
value_label3 = pd.Series(label3)
#ajout des labels aux dataframe
df.insert(loc=1, column='label', value=value_label1)	#ajoute les value_label à la 2e colonne avec le nom de colonne 'label'
df1.insert(loc=1, column='label', value=value_label2)
df2.insert(loc=1, column='label', value=value_label3)

#concatenantion et creation du signal envoye
DATA = pd.concat([df, df, df1, df1, df2, df2],ignore_index=True)


# ***************************************************************************************************************
# 											connection
# ***************************************************************************************************************
#connect fct
def on_connect(client, userdata, flags, rc):
	if rc == 0:
		client.connected_flag = True
		print("Connection OK")
	else:
		print("Echec: ", rc)
		client.loop_stop()


mqtt.Client.connected_flag = False 	#on met le flag à false en attendant la connexion

ip_broker = input("Entrez l'IP du broker : ")	#demande l'IPv4 de la wifi de l'utilisateur taper ipconfig dans un invite de commande
ip_broker = str(ip_broker) #converti l'ip en string pour la fct client

port_broker = 1883 #port mqtt

client = mqtt.Client("python1")	#cree le client avec mqtt
client.on_connect = on_connect	#appel de la fonction on_connect
print("Connection broker: ", ip_broker)

# try expect pour etre sur de la connection
try:
	client.connect(ip_broker, port_broker)
	print("debut de connection")
except:
	print("connexion impossible")
	sys.exit(1)

client.loop_start() #fait une loop 

#loop de connexion, tant que ce n'est pas connecte il print connextion en court
while not client.connected_flag:
	print("Connection en court")
	time.sleep(1)

#en cas de refus de connexion : -verifier que l'on est bien connecte 
#								-verifier que mosquitto.exe a bien ete lance
#								-verifier l'adresse IPv4 dans 
# 								-verifier que le recepteur reçois sur la bonne adresse IP

# ***************************************************************************************************************
# 												envoi
# ***************************************************************************************************************
#init var
i = 0 #index de la boucle for ira de 1 a len(DATA)
valeur = ""	#var comportant les mesures
start = time.time_ns() #debut du temps (au lancement du programme)

#boucle des operations
while True:
	for i in range(len(DATA)):	#remettre data pour envoyer tout le fichier
		valeur = DATA.at[i,'mesure']	#prend la valeur dans ligne d'index i et la colone mesure
		label = DATA.at[i,'label']
		end = time.time_ns()	#temps a l'envoi
		temps = (end - start)/1e9 #temps ecoule en seconde 
		donne = str(temps) + ";" + str(valeur) + ";" + str(label)	#on converti en string et on les additionne separees par un ; pour mieux les traiter a la reception
		
		client.publish("esp32/temperature", donne)	#publie la donnee sur le broker
		
		time.sleep(4e-5)	#pas de temps simule du capteur

		
client.loop_stop()

client.disconnect()
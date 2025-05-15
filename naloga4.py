import cv2 as cv
import numpy as np

def camshift(slika, sablone, iteracije, napaka):
    pass

def zaznaj_gibanje(cap, st_objektov=1):
    '''Implemenacija zaznavanja gibanja, kjer poiščemo število premikajočih objektov glede na parameter st_objektov.'''
    pass

def izracunaj_znacilnice(lokacije_oken, prva_slika):
    '''Izračunaj značilnice za sledenje iz prvega okna.'''
    pass

if __name__ == "__main__":
    #Naloži video
    zaznaj_gibanje = "rocno"
    cap = cv.VideoCapture('video.mp4')
    #Preveri, če je video uspešno naložen
    if not cap.isOpened():
        print("Napaka: Video ne obstaja ali ni bil uspešno naložen.")
        exit(1)

    #Nastavitve meanshift algoritma
    iteracije = 10
    napaka = 1

    lokacije_oken = list()
    #Začetna točka sledenja ročno
    #(x, y, w, h)
    if zaznaj_gibanje == "rocno":
        lokacije_oken.append((0, 0, 0, 0))
        lokacije_oken.append((0, 0, 0, 0))
    else:
        #Začetna točka sledenja avtomatsko
        lokacije_oken = zaznaj_gibanje(cap, st_objektov=2)
    
    #Izračun značilnic za sledenje
    uspel, prva_slika = cap.read()
    sablone = izracunaj_znacilnice(lokacije_oken,prva_slika)


    #Začetek sledenja
    while True:
        uspel, slika = cap.read()
        if not uspel:
            break        
        
        lokacije_novih_oken = camshift(slika, sablone,lokacije_oken, iteracije, napaka)
        lokacije_oken = lokacije_novih_oken
        #Nariši okno
        for okno in lokacije_novih_oken:
            x,y,w,h = okno
            cv.rectangle(slika, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv.imshow('Rezultat', slika)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
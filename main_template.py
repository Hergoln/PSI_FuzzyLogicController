#
# Podstawy Sztucznej Inteligencji, IIS 2020
# Autor: Tomasz Jaworski
# Opis: Szablon kodu do stabilizacji odwróconego wahadła (patyka) w pozycji pionowej podczas ruchu wózka.
#

import gym # Instalacja: https://github.com/openai/gym
import time
from helper import HumanControl, Keys, CartForce
from functions import Display_membership_functions, Compute_weighted_integral, FORCE_DOMAIN
import matplotlib.pyplot as plt


import numpy as np
import skfuzzy as fuzz

#
# przygotowanie środowiska
#   
control = HumanControl()
env = gym.make('gym_PSI:CartPole-v2')
env.reset()
env.render()


def on_key_press(key: int, mod: int):
    global control
    force = 10
    if key == Keys.LEFT:
        control.UserForce = force * CartForce.UNIT_LEFT # krok w lewo
    if key == Keys.RIGHT:
        control.UserForce = force * CartForce.UNIT_RIGHT # krok w prawo
    if key == Keys.P: # pauza
        control.WantPause = True
    if key == Keys.R: # restart
        control.WantReset = True
    if key == Keys.ESCAPE or key == Keys.Q: # wyjście
        control.WantExit = True

env.unwrapped.viewer.window.on_key_press = on_key_press

#########################################################
# KOD INICJUJĄCY - do wypełnienia
#########################################################

"""
1. Określ dziedzinę dla każdej zmiennej lingwistycznej. Każda zmienna ma własną dziedzinę.
2. Zdefiniuj funkcje przynależności dla wybranych przez siebie zmiennych lingwistycznych.
3. Wyświetl je, w celach diagnostycznych.
"""

# cart_position
# cart_velocity

# pole_angle wystarczy do trzymania patyka w górze
# pole_angle # <-Inf, +Inf> w zaleznosci od w ktora strone sie obroci w radianach no i oczywiscie powyze (i ponizej) PI/2 juz nie ma sensu probowac

# te funkcje zapewniają, że kąt jest całokowicie pozytywny lub całkowicie negatywny przy 30 stopniach
DEGREES_DIV = 24
def pole_angle_membership_func_neg(x):
    return min(max(-x, 0) * DEGREES_DIV / np.pi, 1)
def pole_angle_membership_func_zer(x):
    return max((-abs(x)) * DEGREES_DIV / np.pi + 1, 0)
def pole_angle_membership_func_pos(x):
    return min(max(x, 0) * DEGREES_DIV / np.pi, 1)

angles = np.linspace(-np.pi / 6, np.pi / 6, 101)
pole_angle_tup_neg = (angles, pole_angle_membership_func_neg, 'r', 'Negative')
pole_angle_tup_zer = (angles, pole_angle_membership_func_zer, 'g', 'Zero')
pole_angle_tup_pos = (angles, pole_angle_membership_func_pos, 'b', 'Positive')

# tip_velocity
def tip_velocity_func_neg(x):
    return -1
def tip_velocity_func_zer(x):
    return 0
def tip_velocity_func_pos(x):
    return 1



def force_membership_func_neg(x):
    return min(max(-x, 0) / 5, 1)
def force_membership_func_zer(x):
    return max(-abs(x) / 5 + 1, 0)
def force_membership_func_pos(x):
    return min(max(x, 0) / 5, 1)

values = np.linspace(-FORCE_DOMAIN, FORCE_DOMAIN, 101)
force_tup_neg = (values, force_membership_func_neg, 'r', 'Negative')
force_tup_zer = (values, force_membership_func_zer, 'g', 'Zero')
force_tup_pos = (values, force_membership_func_pos, 'b', 'Positive')

Display_membership_functions('Angle', pole_angle_tup_neg, pole_angle_tup_zer, pole_angle_tup_pos)
Display_membership_functions('Force', force_tup_neg, force_tup_zer, force_tup_pos)
plt.show()

#########################################################
# KONIEC KODU INICJUJĄCEGO
#########################################################


#
# Główna pętla symulacji
#
while not control.WantExit:

    #
    # Wstrzymywanie symulacji:
    # Pierwsze wciśnięcie klawisza 'p' wstrzymuje; drugie wciśnięcie 'p' wznawia symulację.
    #
    if control.WantPause:
        control.WantPause = False
        while not control.WantPause:
            time.sleep(0.1)
            env.render()
        control.WantPause = False

    #
    # Czy użytkownik chce zresetować symulację?
    if control.WantReset:
        control.WantReset = False
        env.reset()


    ###################################################
    # ALGORYTM REGULACJI - do wypełnienia
    ##################################################

    """
    Opis wektora stanu (env.state)
        cart_position   -   Położenie wózka w osi X. Zakres: -2.5 do 2.5. Ppowyżej tych granic wózka znika z pola widzenia.
        cart_velocity   -   Prędkość wózka. Zakres +- Inf, jednak wartości powyżej +-2.0 generują zbyt szybki ruch.
        pole_angle      -   Pozycja kątowa patyka, a<0 to odchylenie w lewo, a>0 odchylenie w prawo. Pozycja kątowa ma
                            charakter bezwzględny - do pozycji wliczane są obroty patyka.
                            Ze względów intuicyjnych zaleca się konwersję na stopnie (+-180).
        tip_velocity    -   Prędkość wierzchołka patyka. Zakres +- Inf. a<0 to ruch przeciwny do wskazówek zegara,
                            podczas gdy a>0 to ruch zgodny z ruchem wskazówek zegara.
                            
    Opis zadajnika akcji (fuzzy_response):
        Jest to wartość siły przykładana w każdej chwili czasowej symulacji, wyrażona w Newtonach.
        Zakładany krok czasowy symulacji to env.tau (20 ms).
        Przyłożenie i utrzymanie stałej siły do wózka spowoduje, że ten będzie przyspieszał do nieskończoności,
        ruchem jednostajnym.
    """

    cart_position, cart_velocity, pole_angle, tip_velocity = env.state # Wartości zmierzone


    """
    
    1. Przeprowadź etap rozmywania, w którym dla wartości zmierzonych wyznaczone zostaną ich przynależności do poszczególnych
       zmiennych lingwistycznych. Jedno fizyczne wejście (źródło wartości zmierzonych, np. położenie wózka) posiada własną
       zmienną lingwistyczną.
    """
    u_pole_angle_neg = pole_angle_membership_func_neg(pole_angle)
    u_pole_angle_zer = pole_angle_membership_func_zer(pole_angle)
    u_pole_angle_pos = pole_angle_membership_func_pos(pole_angle)
           
    """
    2. Wyznacza wartości aktywacji reguł rozmytych, wyznaczając stopień ich prawdziwości.       
       Przyjmując, że spójnik LUB (suma rozmyta) to max() a ORAZ/I (iloczyn rozmyty) to min() sprawdź funkcje fmax i fmin.

       dla samego pole_angle:
       (R0)JEŻELI kąt jest ujemny TO siła jest ujemna       
       (R1)JEŻELI kąt jest zerowy TO siła jest zerowa 
       (R2)JEŻELI kąt jest dodatni TO siła jest dodatnia
       
    """
    R0 = u_pole_angle_neg
    R1 = u_pole_angle_zer
    R2 = u_pole_angle_pos

    """
    3. Przeprowadź agregację reguł o tej samej konkluzji.
       Jeżeli masz kilka reguł, posiadających tę samą konkluzję (ale różne przesłanki) to poziom aktywacji tych reguł
       należy agregować tak, aby jedna konkluzja miała jeden poziom aktywacji. Skorzystaj z sumy rozmytej.
    """
    # dla jednej zmiennej pole_angle nie ma tych samych konkluzji
    
    """
    4. Dla każdej reguły przeprowadź operację wnioskowania Mamdaniego.
       Operatorem wnioskowania jest min().
       Przykład: Jeżeli lingwistyczna zmienna wyjściowa ForceToApply ma 5 wartości (strong left, light left, idle, light right, strong right)
       to liczba wyrażeń wnioskujących wyniesie 5 - po jednym wywołaniu operatora Mamdaniego dla konkluzji.
       
       W ten sposób wyznaczasz aktywacje poszczególnych wartości lingwistycznej zmiennej wyjściowej.
       Uważaj - aktywacja wartości zmiennej lingwistycznej w konkluzji to nie liczba a zbiór rozmyty.
       Ponieważ stosujesz operator min(), to wynikiem będzie "przycięty od góry" zbiór rozmyty. 
    """
    u_force_neg_prim = lambda y : min(R0, force_membership_func_neg(y))
    u_force_zer_prim = lambda y : min(R1, force_membership_func_zer(y))
    u_force_pos_prim = lambda y : min(R2, force_membership_func_pos(y))
    
    """
    5. Agreguj wszystkie aktywacje dla danej zmiennej wyjściowej.
    """
    # dla jednej zmiennej pole_angle nie ma tych samych konkluzji
    
    """
    6. Dokonaj defuzyfikacji (np. całkowanie ważone - centroid).
    """
    out_force_func = lambda y : max(u_force_neg_prim(y), u_force_zer_prim(y), u_force_pos_prim(y))
    temp_fuzzy_response = Compute_weighted_integral(out_force_func)
    
    """
    7. Czym będzie wyjściowa wartość skalarna?
    """

    fuzzy_response = temp_fuzzy_response # do zmiennej fuzzy_response zapisz wartość siły, jaką chcesz przyłożyć do wózka.
    #
    # KONIEC algorytmu regulacji
    #########################

    # Jeżeli użytkownik chce przesunąć wózek, to jego polecenie ma wyższy priorytet
    if control.UserForce is not None:
        applied_force = control.UserForce
        control.UserForce = None
    else:
        applied_force = fuzzy_response

    #
    # Wyświetl stan środowiska oraz wartość odpowiedzi regulatora na ten stan.
    print(
        f"cpos={cart_position:8.4f}, cvel={cart_velocity:8.4f}, pang={pole_angle:8.4f}, tvel={tip_velocity:8.4f}, force={applied_force:8.4f}")

    #
    # Wykonaj krok symulacji
    env.step(applied_force)

    #
    # Pokaż kotku co masz w środku
    env.render()

#
# Zostaw ten patyk!
env.close()


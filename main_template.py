#
# Podstawy Sztucznej Inteligencji, IIS 2020
# Autor: Tomasz Jaworski
# Opis: Szablon kodu do stabilizacji odwróconego wahadła (patyka) w pozycji pionowej podczas ruchu wózka.
#

import gym # Instalacja: https://github.com/openai/gym
import time
import sys
from helper import HumanControl, Keys, CartForce
from functions import * #Display_membership_functions, Compute_weighted_integral_force, FORCE_DOMAIN
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

given = 0 # value added to position to simulate position shift
CART_POSITION_VALUES_RANGE = 0.1
CART_POSITION_DISPLAY_RANGE = CART_POSITION_VALUES_RANGE * 3

DEGREES_DIV = 60
DEGREES_DISPLAY_RANGE = np.pi / 6.0

FORCE_VALUE_RANGE = 10
FORCE_VALUE_DISPLAY_RANGE = FORCE_VALUE_RANGE + 2

# cart_position
cart_position_funcs = Generic_membership_functions(CART_POSITION_VALUES_RANGE)
Display_membership_functions(
    'Cart position',
    CART_POSITION_DISPLAY_RANGE,
    cart_position_funcs[NEGATIVE],
    cart_position_funcs[ZERO],
    cart_position_funcs[POSITIVE],
    )

# pole_angle 
pole_angle_funcs = Generic_membership_functions(np.pi / DEGREES_DIV)
Display_membership_functions(
    'Angle',
    DEGREES_DISPLAY_RANGE,
    pole_angle_funcs[NEGATIVE],
    pole_angle_funcs[ZERO],
    pole_angle_funcs[POSITIVE],
    )

# force
force_funcs = Generic_membership_functions(FORCE_VALUE_RANGE)
Display_membership_functions(
    'Force',
    FORCE_VALUE_DISPLAY_RANGE,
    force_funcs[NEGATIVE],
    force_funcs[ZERO],
    force_funcs[POSITIVE],
    )

if len(sys.argv) > 1 and sys.argv[1] is 's':
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
        print("=RESET=")
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
    ang_neg = pole_angle_funcs[NEGATIVE](pole_angle)
    ang_zer = pole_angle_funcs[ZERO](pole_angle)
    ang_pos = pole_angle_funcs[POSITIVE](pole_angle)
    print(f"ang neg: {ang_neg:8.4f}, ang zer: {ang_zer:8.4f}, ang pos: {ang_pos:8.4f}")
    
    cart_position = cart_position - given
    pos_neg = cart_position_funcs[NEGATIVE](cart_position)
    pos_zer = cart_position_funcs[ZERO](cart_position)
    pos_pos = cart_position_funcs[POSITIVE](cart_position)
    print(f"pos neg: {pos_neg:8.4f}, pos zer: {pos_zer:8.4f}, pos pos: {pos_pos:8.4f}")
    

    """
    2. Wyznacza wartości aktywacji reguł rozmytych, wyznaczając stopień ich prawdziwości.       
       Przyjmując, że spójnik LUB (suma rozmyta) to max() a ORAZ/I (iloczyn rozmyty) to min() sprawdź funkcje fmax i fmin.
       (kąt, pr_t, pos, pr_c)
       
       (R0) IF kąt neg TO s neg
       (R1) IF kąt pos TO s pos
       
       (R5) IF kąt zer AND pos neg TO s pos
       (R6) IF kąt zer AND pos pos TO s neg

    """
    R0 = ang_neg
    R1 = ang_pos

    R5 = min(ang_zer, pos_pos)
    R6 = min(ang_zer, pos_neg)

    """
    3. Przeprowadź agregację reguł o tej samej konkluzji.
       Jeżeli masz kilka reguł, posiadających tę samą konkluzję (ale różne przesłanki) to poziom aktywacji tych reguł
       należy agregować tak, aby jedna konkluzja miała jeden poziom aktywacji. Skorzystaj z sumy rozmytej.
    """
    unified_neg_rule = max(R0, R6)
    unified_zer_rule = 0
    unified_pos_rule = max(R1, R5)
    
    """
    4. Dla każdej reguły przeprowadź operację wnioskowania Mamdaniego.
       Operatorem wnioskowania jest min().
       Przykład: Jeżeli lingwistyczna zmienna wyjściowa ForceToApply ma 5 wartości (strong left, light left, idle, light right, strong right)
       to liczba wyrażeń wnioskujących wyniesie 5 - po jednym wywołaniu operatora Mamdaniego dla konkluzji.
       
       W ten sposób wyznaczasz aktywacje poszczególnych wartości lingwistycznej zmiennej wyjściowej.
       Uważaj - aktywacja wartości zmiennej lingwistycznej w konkluzji to nie liczba a zbiór rozmyty.
       Ponieważ stosujesz operator min(), to wynikiem będzie "przycięty od góry" zbiór rozmyty. 
    """
    u_force_neg_prim = lambda y : min(unified_neg_rule, force_funcs[NEGATIVE](y))
    u_force_zer_prim = lambda y : min(unified_zer_rule, force_funcs[ZERO](y))
    u_force_pos_prim = lambda y : min(unified_pos_rule, force_funcs[POSITIVE](y))
    
    
    """
    5. Agreguj wszystkie aktywacje dla danej zmiennej wyjściowej.
    """
    # dla jednej zmiennej pole_angle nie ma tych samych konkluzji
    
    """
    6. Dokonaj defuzyfikacji (np. całkowanie ważone - centroid).
    """
    out_force_func = lambda y : max(u_force_neg_prim(y), u_force_zer_prim(y), u_force_pos_prim(y))
    temp_fuzzy_response = Compute_weighted_integral_force(out_force_func, FORCE_VALUE_RANGE)
    
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
        f"cpos={cart_position:8.4f}, cvel={cart_velocity:8.4f}, pang={pole_angle:8.4f}, tvel={tip_velocity:8.4f}, force={fuzzy_response:8.4f}\n")

    #
    # Wykonaj krok symulacji
    env.step(applied_force)

    #
    # Pokaż kotku co masz w środku
    env.render()

#
# Zostaw ten patyk!
env.close()


#
# Podstawy Sztucznej Inteligencji, IIS 2020
# Autor: Tomasz Jaworski
# Opis: Szablon kodu do stabilizacji odwróconego wahadła (patyka) w pozycji pionowej podczas ruchu wózka.
#

import gym # Instalacja: https://github.com/openai/gym
import time
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


given = 0
CART_POSITION_VALUES_RANGE = 0.2
CART_POSITION_DISPLAY_RANGE = CART_POSITION_VALUES_RANGE + 2

CART_VELOCITY_VALUES_RANGE = 1.0
CART_VELOCITY_DISPLAY_RANGE = CART_VELOCITY_VALUES_RANGE + 1.0

DEGREES_DIV = 12.0
DEGREES_DISPLAY_RANGE = np.pi / 6.0

TIP_VELOCITY_VALUES_RANGE = 0.2
TIP_VELOCITY_DISPLAY_RANGE = TIP_VELOCITY_VALUES_RANGE + 2

FORCE_VALUE_RANGE = 12
FORCE_VALUE_DISPLAY_RANGE = FORCE_VALUE_RANGE + 2
# cart_position
cart_position_funcs = Generic_membership_functions(CART_POSITION_VALUES_RANGE - given)
Display_membership_functions(
    'Cart position',
    CART_POSITION_DISPLAY_RANGE,
    cart_position_funcs[NEGATIVE],
    cart_position_funcs[ZERO],
    cart_position_funcs[POSITIVE],
    )

# cart_velocity
cart_velocity_funcs = Generic_membership_functions(CART_VELOCITY_VALUES_RANGE)
Display_membership_functions(
    'Cart velocity',
    CART_VELOCITY_DISPLAY_RANGE,
    cart_velocity_funcs[NEGATIVE],
    cart_velocity_funcs[ZERO],
    cart_velocity_funcs[POSITIVE],
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

# tip_velocity
tip_velocities_funcs = Generic_membership_functions(TIP_VELOCITY_VALUES_RANGE)
Display_membership_functions(
    'Tip velocity',
    TIP_VELOCITY_DISPLAY_RANGE,
    tip_velocities_funcs[NEGATIVE],
    tip_velocities_funcs[ZERO],
    tip_velocities_funcs[POSITIVE],
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

#plt.show()

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
    tip_v_neg = tip_velocities_funcs[NEGATIVE](tip_velocity)
    tip_v_zer = tip_velocities_funcs[ZERO](tip_velocity)
    tip_v_pos = tip_velocities_funcs[POSITIVE](tip_velocity)
    print(f"tiv neg: {tip_v_neg:8.4f}, tiv zer: {tip_v_zer:8.4f}, tiv pos: {tip_v_pos:8.4f}")
    cart_v_neg = cart_velocity_funcs[NEGATIVE](cart_velocity)
    cart_v_zer = cart_velocity_funcs[ZERO](cart_velocity)
    cart_v_pos = cart_velocity_funcs[POSITIVE](cart_velocity)
    print(f"cav neg: {cart_v_neg:8.4f}, cav zer: {cart_v_zer:8.4f}, cav pos: {cart_v_pos:8.4f}")
    pos_neg = cart_position_funcs[NEGATIVE](cart_position)
    pos_zer = cart_position_funcs[ZERO](cart_position)
    pos_pos = cart_position_funcs[POSITIVE](cart_position)
    print(f"pos neg: {pos_neg:8.4f}, pos zer: {pos_zer:8.4f}, pos pos: {pos_pos:8.4f}")
    

    """
    2. Wyznacza wartości aktywacji reguł rozmytych, wyznaczając stopień ich prawdziwości.       
       Przyjmując, że spójnik LUB (suma rozmyta) to max() a ORAZ/I (iloczyn rozmyty) to min() sprawdź funkcje fmax i fmin.
       TEORETYCZNIE POWINIENEM DODAĆ 54 REGUŁY
       (kąt, pr_t, pos, pr_c)
       (R0) IF kąt neg I pr_t neg I pos neg I pr_c neg TO s neg
       (R1) IF kąt neg I pr_t neg I pos neg I pr_c zer TO s neg
       (R2) IF kąt neg I pr_t neg I pos neg I pr_c pos TO s neg
       
       (R3) IF kąt neg I pr_t neg I pos zer I pr_c neg TO s neg
       (R4) IF kąt neg I pr_t neg I pos zer I pr_c zer TO s neg
       (R5) IF kąt neg I pr_t neg I pos zer I pr_c pos TO s neg
       
       (R6) IF kąt neg I pr_t neg I pos pos I pr_c neg TO s neg
       (R7) IF kąt neg I pr_t neg I pos pos I pr_c zer TO s neg
       (R8) IF kąt neg I pr_t neg I pos pos I pr_c pos TO s neg
       
       #nie jestem przekonany co do pr_t = zer ====
       (R9) IF kąt neg I pr_t zer I pos neg I pr_c neg TO s zer
       (R10)IF kąt neg I pr_t zer I pos neg I pr_c zer TO s neg
       (R11)IF kąt neg I pr_t zer I pos neg I pr_c pos TO s neg
       
       (R12)IF kąt neg I pr_t zer I pos zer I pr_c neg TO s neg
       (R13)IF kąt neg I pr_t zer I pos zer I pr_c zer TO s neg
       (R14)IF kąt neg I pr_t zer I pos zer I pr_c pos TO s neg
       
       (R15)IF kąt neg I pr_t zer I pos pos I pr_c neg TO s neg
       (R16)IF kąt neg I pr_t zer I pos pos I pr_c zer TO s neg
       (R17)IF kąt neg I pr_t zer I pos pos I pr_c pos TO s neg
       #====
       
       (R18)IF kąt neg I pr_t pos I pos neg I pr_c neg TO s pos
       (R19)IF kąt neg I pr_t pos I pos neg I pr_c zer TO s pos
       (R20)IF kąt neg I pr_t pos I pos neg I pr_c pos TO s zer
       
       (R21)IF kąt neg I pr_t pos I pos zer I pr_c neg TO s zer
       (R22)IF kąt neg I pr_t pos I pos zer I pr_c zer TO s zer
       (R23)IF kąt neg I pr_t pos I pos zer I pr_c pos TO s neg
       
       (R24)IF kąt neg I pr_t pos I pos pos I pr_c neg TO s neg
       (R25)IF kąt neg I pr_t pos I pos pos I pr_c zer TO s neg
       (R26)IF kąt neg I pr_t pos I pos pos I pr_c pos TO s neg
       
       
       (R27)IF kąt zer I pos neg I pr_c neg TO s pos
       (R28)IF kąt zer I pos neg I pr_c zer TO s pos
       (R29)IF kąt zer I pos neg I pr_c pos TO s zer
       
       (R30)IF kąt zer I pos zer I pr_c neg TO s pos
       (R31)IF kąt zer I pr_t neg I pos zer I pr_c zer TO s pos
       (R63)IF kąt zer I pr_t zer I pos zer I pr_c zer TO s zer
       (R64)IF kąt zer I pr_t pos I pos zer I pr_c zer TO s neg
       (R32)IF kąt zer I pos zer I pr_c pos TO s neg
       
       (R33)IF kąt zer I pos pos I pr_c neg TO s zer
       (R34)IF kąt zer I pos pos I pr_c zer TO s neg
       (R35)IF kąt zer I pos pos I pr_c pos TO s neg
       
       
       (R36)IF kąt pos I pr_t neg I pos neg I pr_c neg TO s pos
       (R37)IF kąt pos I pr_t neg I pos neg I pr_c zer TO s pos
       (R38)IF kąt pos I pr_t neg I pos neg I pr_c pos TO s pos
       
       (R39)IF kąt pos I pr_t neg I pos zer I pr_c neg TO s pos
       (R40)IF kąt pos I pr_t neg I pos zer I pr_c zer TO s zer
       (R41)IF kąt pos I pr_t neg I pos zer I pr_c pos TO s zer
       
       (R42)IF kąt pos I pr_t neg I pos pos I pr_c neg TO s zer
       (R43)IF kąt pos I pr_t neg I pos pos I pr_c zer TO s neg
       (R44)IF kąt pos I pr_t neg I pos pos I pr_c pos TO s neg
       
       (R45)IF kąt pos I pr_t zer I pos neg I pr_c neg TO s pos
       (R46)IF kąt pos I pr_t zer I pos neg I pr_c zer TO s pos
       (R47)IF kąt pos I pr_t zer I pos neg I pr_c pos TO s pos
       
       (R48)IF kąt pos I pr_t zer I pos zer I pr_c neg TO s pos
       (R49)IF kąt pos I pr_t zer I pos zer I pr_c zer TO s pos
       (R50)IF kąt pos I pr_t zer I pos zer I pr_c pos TO s pos
       
       (R51)IF kąt pos I pr_t zer I pos pos I pr_c neg TO s pos
       (R52)IF kąt pos I pr_t zer I pos pos I pr_c zer TO s pos
       (R53)IF kąt pos I pr_t zer I pos pos I pr_c pos TO s zer
       
       (R54)IF kąt pos I pr_t pos I pos neg I pr_c neg TO s pos
       (R55)IF kąt pos I pr_t pos I pos neg I pr_c zer TO s pos
       (R56)IF kąt pos I pr_t pos I pos neg I pr_c pos TO s pos
       
       (R57)IF kąt pos I pr_t pos I pos zer I pr_c neg TO s pos
       (R58)IF kąt pos I pr_t pos I pos zer I pr_c zer TO s pos
       (R59)IF kąt pos I pr_t pos I pos zer I pr_c pos TO s pos
       
       (R60)IF kąt pos I pr_t pos I pos pos I pr_c neg TO s pos
       (R61)IF kąt pos I pr_t pos I pos pos I pr_c zer TO s pos
       (R62)IF kąt pos I pr_t pos I pos pos I pr_c pos TO s pos
    """
    
    R0 = min(ang_neg, tip_v_neg, pos_neg, cart_v_neg) #zer
    R1 = min(ang_neg, tip_v_neg, pos_neg, cart_v_zer) #neg
    R2 = min(ang_neg, tip_v_neg, pos_neg, cart_v_pos) #neg
    
    R3 = min(ang_neg, tip_v_neg, pos_zer, cart_v_neg) #zer
    R4 = min(ang_neg, tip_v_neg, pos_zer, cart_v_zer) #neg
    R5 = min(ang_neg, tip_v_neg, pos_zer, cart_v_pos) #neg
    
    R6 = min(ang_neg, tip_v_neg, pos_pos, cart_v_neg) #neg
    R7 = min(ang_neg, tip_v_neg, pos_pos, cart_v_zer) #neg
    R8 = min(ang_neg, tip_v_neg, pos_pos, cart_v_pos) #neg
    
    R9 = min(ang_neg, tip_v_zer, pos_neg, cart_v_neg) #meg
    R10 = min(ang_neg, tip_v_zer, pos_neg, cart_v_zer) #neg
    R11 = min(ang_neg, tip_v_zer, pos_neg, cart_v_pos) #neg
    
    R12 = min(ang_neg, tip_v_zer, pos_zer, cart_v_neg) #neg
    R13 = min(ang_neg, tip_v_zer, pos_zer, cart_v_zer) #neg
    R14 = min(ang_neg, tip_v_zer, pos_zer, cart_v_pos) #neg
    
    R15 = min(ang_neg, tip_v_zer, pos_pos, cart_v_neg) #neg
    R16 = min(ang_neg, tip_v_zer, pos_pos, cart_v_zer) #neg
    R17 = min(ang_neg, tip_v_zer, pos_pos, cart_v_pos) #neg
    
    R18 = min(ang_neg, tip_v_pos, pos_neg, cart_v_neg) #pos
    R19 = min(ang_neg, tip_v_pos, pos_neg, cart_v_zer) #pos
    R20 = min(ang_neg, tip_v_pos, pos_neg, cart_v_pos) #zer
    
    R21 = min(ang_neg, tip_v_pos, pos_zer, cart_v_neg) #zer
    R22 = min(ang_neg, tip_v_pos, pos_zer, cart_v_zer) #zer
    R23 = min(ang_neg, tip_v_pos, pos_zer, cart_v_pos) #neg
    
    R24 = min(ang_neg, tip_v_pos, pos_pos, cart_v_neg) #neg
    R25 = min(ang_neg, tip_v_pos, pos_pos, cart_v_zer) #neg
    R26 = min(ang_neg, tip_v_pos, pos_pos, cart_v_pos) #neg
    
    R27 = min(ang_zer, pos_neg, cart_v_neg) #pos
    R28 = min(ang_zer, pos_neg, cart_v_zer) #pos
    R29 = min(ang_zer, pos_neg, cart_v_pos) #pos
    
    R30 = min(ang_zer, pos_zer, cart_v_neg) #pos
    R31 = min(ang_zer, tip_v_neg, pos_zer, cart_v_zer) #pos
    R63 = min(ang_zer, tip_v_zer, pos_zer, cart_v_zer) #zer
    R64 = min(ang_zer, tip_v_pos, pos_zer, cart_v_zer) #neg
    R32 = min(ang_zer, pos_zer, cart_v_pos) #neg
    
    R33 = min(ang_zer, pos_pos, cart_v_neg) #neg
    R34 = min(ang_zer, pos_pos, cart_v_zer) #neg
    R35 = min(ang_zer, pos_pos, cart_v_pos) #neg
    
    R36 = min(ang_pos, tip_v_neg, pos_neg, cart_v_neg) #pos
    R37 = min(ang_pos, tip_v_neg, pos_neg, cart_v_zer) #pos
    R38 = min(ang_pos, tip_v_neg, pos_neg, cart_v_pos) #pos
    
    R39 = min(ang_pos, tip_v_neg, pos_zer, cart_v_neg) #pos
    R40 = min(ang_pos, tip_v_neg, pos_zer, cart_v_zer) #zer
    R41 = min(ang_pos, tip_v_neg, pos_zer, cart_v_pos) #zer
    
    R42 = min(ang_pos, tip_v_neg, pos_pos, cart_v_neg) #zer
    R43 = min(ang_pos, tip_v_neg, pos_pos, cart_v_zer) #neg
    R44 = min(ang_pos, tip_v_neg, pos_pos, cart_v_pos) #neg
    
    R45 = min(ang_pos, tip_v_zer, pos_neg, cart_v_neg) #pos
    R46 = min(ang_pos, tip_v_zer, pos_neg, cart_v_zer) #pos
    R47 = min(ang_pos, tip_v_zer, pos_neg, cart_v_pos) #pos
    
    R48 = min(ang_pos, tip_v_zer, pos_zer, cart_v_neg) #pos
    R49 = min(ang_pos, tip_v_zer, pos_zer, cart_v_zer) #pos
    R50 = min(ang_pos, tip_v_zer, pos_zer, cart_v_pos) #pos
    
    R51 = min(ang_pos, tip_v_zer, pos_pos, cart_v_neg) #pos
    R52 = min(ang_pos, tip_v_zer, pos_pos, cart_v_zer) #pos
    R53 = min(ang_pos, tip_v_zer, pos_pos, cart_v_pos) #pos
    
    R54 = min(ang_pos, tip_v_pos, pos_neg, cart_v_neg) #pos
    R55 = min(ang_pos, tip_v_pos, pos_neg, cart_v_zer) #pos
    R56 = min(ang_pos, tip_v_pos, pos_neg, cart_v_pos) #pos
    
    R57 = min(ang_pos, tip_v_pos, pos_zer, cart_v_neg) #pos
    R58 = min(ang_pos, tip_v_pos, pos_zer, cart_v_zer) #pos
    R59 = min(ang_pos, tip_v_pos, pos_zer, cart_v_pos) #zer
    
    R60 = min(ang_pos, tip_v_pos, pos_pos, cart_v_neg) #pos
    R61 = min(ang_pos, tip_v_pos, pos_pos, cart_v_zer) #pos
    R62 = min(ang_pos, tip_v_pos, pos_pos, cart_v_pos) #zer
    
    """
    3. Przeprowadź agregację reguł o tej samej konkluzji.
       Jeżeli masz kilka reguł, posiadających tę samą konkluzję (ale różne przesłanki) to poziom aktywacji tych reguł
       należy agregować tak, aby jedna konkluzja miała jeden poziom aktywacji. Skorzystaj z sumy rozmytej.
    """
    unified_neg_rule = max(R1, R2,
                           R4, R5, R6, R7,
                           R8, R9, R10, R11, R12,
                           R13, R14, R15, R16,
                           R17, R23, R24, R25,
                           R26, R29, R32, R34,
                           R35, R43, R44, R64)
    print(f"unified_neg_rule = {unified_neg_rule}")
    unified_zer_rule = max(R0, R3, R20, R21,
                           R22, R40, R41, R42,
                           R59, R63, R62)
    print(f"unified_zer_rule = {unified_zer_rule}")
    unified_pos_rule = max(R18, R19, R27, R28,
                           R30, R31, R33, R36,
                           R37, R38, R39, R45,
                           R46, R47, R48, R49,
                           R50, R51, R52, R53,
                           R54, R55, R56, R57,
                           R58, R60, R61)
    print(f"unified_pos_rule = {unified_pos_rule}")
    
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
    temp_fuzzy_response = Compute_weighted_integral_force(out_force_func)
    
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
        f"cpos={cart_position:8.4f}, cvel={cart_velocity:8.4f}, pang={pole_angle:8.4f}, tvel={tip_velocity:8.4f}, force={applied_force:8.4f}\n")

    #
    # Wykonaj krok symulacji
    env.step(applied_force)

    #
    # Pokaż kotku co masz w środku
    env.render()

#
# Zostaw ten patyk!
env.close()


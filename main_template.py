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

# cart_position
given = 0
CART_POSITION_VALUES_RANGE = 0.5
CART_POSITION_DISPLAY_RANGE = CART_POSITION_VALUES_RANGE + 2
cart_position_funcs = Generic_membership_functions(CART_POSITION_VALUES_RANGE - given)
Display_membership_functions(
    'Cart position',
    CART_POSITION_DISPLAY_RANGE,
    cart_position_funcs[NEGATIVE],
    cart_position_funcs[ZERO],
    cart_position_funcs[POSITIVE],
    )

# cart_velocity
CART_VELOCITY_VALUES_RANGE = 0.2
CART_VELOCITY_DISPLAY_RANGE = CART_VELOCITY_VALUES_RANGE + 1.0
cart_velocity_funcs = Generic_membership_functions(CART_VELOCITY_VALUES_RANGE)
Display_membership_functions(
    'Cart velocity',
    CART_VELOCITY_DISPLAY_RANGE,
    cart_velocity_funcs[NEGATIVE],
    cart_velocity_funcs[ZERO],
    cart_velocity_funcs[POSITIVE],
    )

# pole_angle 
DEGREES_DIV = 24.0
DEGREES_DISPLAY_RANGE = np.pi / 6.0
pole_angle_funcs = Generic_membership_functions(np.pi / DEGREES_DIV)
Display_membership_functions(
    'Angle',
    DEGREES_DISPLAY_RANGE,
    pole_angle_funcs[NEGATIVE],
    pole_angle_funcs[ZERO],
    pole_angle_funcs[POSITIVE],
    )

# tip_velocity
TIP_VELOCITY_VALUES_RANGE = 0.5
TIP_VELOCITY_DISPLAY_RANGE = TIP_VELOCITY_VALUES_RANGE + 2
tip_velocities_funcs = Generic_membership_functions(TIP_VELOCITY_VALUES_RANGE)
Display_membership_functions(
    'Tip velocity',
    TIP_VELOCITY_DISPLAY_RANGE,
    tip_velocities_funcs[NEGATIVE],
    tip_velocities_funcs[ZERO],
    tip_velocities_funcs[POSITIVE],
    )

# force
FORCE_VALUE_RANGE = FORCE_DOMAIN
FORCE_VALUE_DISPLAY_RANGE = FORCE_VALUE_RANGE + 2
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
    u_cart_position_neg = cart_position_funcs[NEGATIVE](cart_position)
    u_cart_position_zer = cart_position_funcs[ZERO](cart_position)
    u_cart_position_pos = cart_position_funcs[POSITIVE](cart_position)
    print(f"pos neg: {u_cart_position_neg}, pos zer: {u_cart_position_zer}, pos pos: {u_cart_position_pos}")
    u_cart_velocity_neg = cart_velocity_funcs[NEGATIVE](cart_velocity)
    u_cart_velocity_zer = cart_velocity_funcs[ZERO](cart_velocity)
    u_cart_velocity_pos = cart_velocity_funcs[POSITIVE](cart_velocity)
    
    u_pole_angle_neg = pole_angle_funcs[NEGATIVE](pole_angle)
    u_pole_angle_zer = pole_angle_funcs[ZERO](pole_angle)
    u_pole_angle_pos = pole_angle_funcs[POSITIVE](pole_angle)
    print(f"ang neg: {u_pole_angle_neg}, ang zer: {u_pole_angle_zer}, ang pos: {u_pole_angle_pos}")
    u_tip_velocity_neg = tip_velocities_funcs[NEGATIVE](tip_velocity)
    u_tip_velocity_zer = tip_velocities_funcs[ZERO](tip_velocity)
    u_tip_velocity_pos = tip_velocities_funcs[POSITIVE](tip_velocity)
    print(f"tip_v neg: {u_tip_velocity_neg}, tip_v zer: {u_tip_velocity_zer}, tip_v pos: {u_tip_velocity_pos}")

    """
    2. Wyznacza wartości aktywacji reguł rozmytych, wyznaczając stopień ich prawdziwości.       
       Przyjmując, że spójnik LUB (suma rozmyta) to max() a ORAZ/I (iloczyn rozmyty) to min() sprawdź funkcje fmax i fmin.

       dla samego pole_angle:
       (R0)JEŻELI kąt jest ujemny TO siła jest ujemn {R0 = u_pole_angle_neg} 
       (R1)JEŻELI kąt jest zerowy TO siła jest zerowa {R1 = u_pole_angle_zer}
       (R2)JEŻELI kąt jest dodatni TO siła jest dodatnia {R2 = u_pole_angle_pos}
       
       
       
       (pos, pr_c, kąt, pr_t)
       dla zadania utrzymania kąta oraz pozycji (stabilnosc ma priorytet):
           nie interesują nas sytuacje "kąt ujemny I pr_t ujemna I pr_c dodatnia" oraz "kąt dodatni I pr_t dodatnia I pr_c ujemna" <- nie da się z nich uciec
       (R0)JEŻELI kąt ujemny I pr_t ujemna I pr_c zero TO siła ujemna
       (R1)JEŻELI kąt ujemny I pr_t ujemna I pr_c ujemna TO siła ujemna
       
       (R2)JEŻELI kąt ujemny I pr_t zero TO siła ujemna
       
       ## kiedy kij jest w stanie stabilizującym się to zajmujemy się pozycją
       ## to może być kłopotliwe bo upraszczamy pewne sytuacje (np. "kąt dodatni I pr_t ujemna I pr_c dodatnia TO siła dodatnia" jest tutaj tym samym co
                                                                    "kąt ujemny I pr_t dodatnia I pr_c dodatnia TO siła dodatnia" co jest lekkim wyrostem)
       JEŻELI ((kąt ujemny I pr_t dodatnia) LUB (kąt zero I pr_t zero) LUB (kąt dodatni I pr_t ujemna)):
           (R3)I pr_c ujemna I pos ujemna TO siła dodatnia
           (R4)I pr_c ujemna I pos zero TO siła dodatnia
           (R5)I pr_c ujemna I pos dodatnia TO siła zero           
           
           (R6)I pr_c zero I pos ujemna TO siła dodatnia           
           (R7)I pr_c zero I pos zero TO siła zero
           (R8)I pr_c zero I pos dodatnia TO siła ujemna
           
           (R9)I pr_c dodatnia I pos ujemna TO siła zero
           (R10)I pr_c dodatnia I pos zero TO siła ujemna
           (R11)I pr_c dodatnia I pos dodatnia TO siła ujemna
       
       (R12)JEŻELI kąt zero I pr_t ujemna TO siła dodatnia

       (R13)JEŻELI kąt zero I pr_t dodatnia TO siła ujemna
       
       (R14)JEŻELI kąt dodatni I pr_t zero TO siła dodatnia
       
       (R15)JEŻELI kąt dodatni I pr_t dodatnia I pr_c zero TO siła dodatnia
       (R16)JEŻELI kąt dodatni I pr_t dodatnia I pr_c dodatnia TO siła dodatnia
       
       
       
       (pos, kąt, pr_t)
       dla zadania utrzymania kąta oraz pozycji (stabilnosc ma priorytet):
       (R0)JEŻELI kąt ujemny I pr_t ujemna TO siła ujemna
       (R1)JEŻELI kąt ujemny I pr_t zero TO siła ujemna
       
       JEŻELI ((kąt ujemny I pr_t dodatnia) LUB (kąt zero I pr_t zero) LUB (kąt dodatni I pr_t ujemna)):
           (R2)I pos ujemna TO siła dodatnia
           (R3)I pos zero TO siła zero
           (R4)I pos dodatnia TO siła ujemna
           
       (R5)JEŻELI kąt zero I pos ujemna TO siła dodatnia
       (R6)JEŻELI kąt zero I pos zero TO siła zero
       (R7)JEŻELI kąt zero I pos dodatnia TO siła ujemna
       
       (R8)JEŻELI kąt dodatni I pr_t zero TO siła dodatnia
       (R9)JEŻELI kąt dodatni I pr_t dodatnia TO siła dodatnia
       
       (R10)JEŻELI pr_c ujemna TO siła dodatnia
       (R11)JEŻELI pr_c zero TO siła zero
       (R12)JEŻELI pr_c dodatnia TO siła ujemna
    """
    R0 = min(u_pole_angle_neg, u_tip_velocity_neg) # neg
    R1 = min(u_pole_angle_neg, u_tip_velocity_zer) # neg
    R2 = min(                                       # pos
            max(
                min(u_pole_angle_neg, u_tip_velocity_pos),
                min(u_pole_angle_zer, u_tip_velocity_zer),
                min(u_pole_angle_pos, u_tip_velocity_neg)
                ),
            u_cart_position_neg
            ) 
    R3 = min(                                       # zer
            max(
                min(u_pole_angle_neg, u_tip_velocity_pos),
                min(u_pole_angle_zer, u_tip_velocity_zer),
                min(u_pole_angle_pos, u_tip_velocity_neg)
                ),
            u_cart_position_zer
            ) 
    R4 = min(                                       # neg
            max(
                min(u_pole_angle_neg, u_tip_velocity_pos),
                min(u_pole_angle_zer, u_tip_velocity_zer),
                min(u_pole_angle_pos, u_tip_velocity_neg)
                ),
            u_cart_position_neg
            ) 
    R5 = min(u_pole_angle_zer, u_cart_position_neg) # pos
    R6 = min(u_pole_angle_zer, u_cart_position_zer) # zer
    R7 = min(u_pole_angle_zer, u_cart_position_pos) # neg
    R8 = min(u_pole_angle_pos, u_tip_velocity_zer) # pos
    R9 = min(u_pole_angle_pos, u_tip_velocity_pos) # pos

    """
    3. Przeprowadź agregację reguł o tej samej konkluzji.
       Jeżeli masz kilka reguł, posiadających tę samą konkluzję (ale różne przesłanki) to poziom aktywacji tych reguł
       należy agregować tak, aby jedna konkluzja miała jeden poziom aktywacji. Skorzystaj z sumy rozmytej.
    """
    unified_neg_rule = max(R0, R1, R4, R7)
    #print(f"unified_neg_rule = {unified_neg_rule}")
    unified_zer_rule = max(R3, R6)
    #print(f"unified_zer_rule = {unified_zer_rule}")
    unified_pos_rule = max(R2, R5, R8, R9)
    #print(f"unified_pos_rule = {unified_pos_rule}")
    
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


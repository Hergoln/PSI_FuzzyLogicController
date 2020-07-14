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
given = 1
CART_POSITION_VALUES_RANGE = 2.5
def cart_position_func_neg(x):
    return min(max(-(x - given), 0) / CART_POSITION_VALUES_RANGE, 1)
def cart_position_func_zer(x):
    return max(-abs((x - given)) / CART_POSITION_VALUES_RANGE + 1, 0)
def cart_position_func_pos(x):
    return min(max((x - given), 0) / CART_POSITION_VALUES_RANGE, 1)

cart_positions_domains = np.linspace(-CART_POSITION_VALUES_RANGE - 1, CART_POSITION_VALUES_RANGE + 1, 101)
cart_position_tups = Memebership_display_tuples(
    cart_positions_domains,
    cart_position_func_neg,
    cart_position_func_zer,
    cart_position_func_pos,
    )
Display_membership_functions('Cart position', *cart_position_tups)

# cart_velocity
CART_VELOCITY_VALUES_RANGE = 2
def cart_velocity_func_neg(x):
    return min(max(-x, 0) / CART_VELOCITY_VALUES_RANGE, 1)
def cart_velocity_func_zer(x):
    return max(-abs(x) / CART_VELOCITY_VALUES_RANGE + 1, 0)
def cart_velocity_func_pos(x):
    return min(max(x, 0) / CART_VELOCITY_VALUES_RANGE, 1)

cart_velocities_domain = np.linspace(-CART_VELOCITY_VALUES_RANGE - 1, CART_VELOCITY_VALUES_RANGE + 1, 101)
cart_velocity_tups = Memebership_display_tuples(
    cart_velocities_domain,
    cart_velocity_func_neg,
    cart_velocity_func_zer,
    cart_velocity_func_pos,
    )
Display_membership_functions('Cart velocity', *cart_velocity_tups)
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
pole_angle_tups = Memebership_display_tuples(
    angles,
    pole_angle_membership_func_neg,
    pole_angle_membership_func_zer,
    pole_angle_membership_func_pos,
    )
Display_membership_functions('Angle', *pole_angle_tups)

# tip_velocity
TIP_VELOCITY_VALUES_RANGE = 2
TIP_VELOCITY_DOMAIN = 5
def tip_velocity_func_neg(x):
    return min(max(-x, 0) / TIP_VELOCITY_VALUES_RANGE, 1)
def tip_velocity_func_zer(x):
    return max(-abs(x) / TIP_VELOCITY_VALUES_RANGE + 1, 0)
def tip_velocity_func_pos(x):
    return min(max(x, 0) / TIP_VELOCITY_VALUES_RANGE, 1)

tip_velocities_domain = np.linspace(-TIP_VELOCITY_DOMAIN, TIP_VELOCITY_DOMAIN, 101)
tip_velocities_tups = Memebership_display_tuples(
    tip_velocities_domain,
    tip_velocity_func_neg,
    tip_velocity_func_zer,
    tip_velocity_func_pos,
    ) 
Display_membership_functions('Tip velocity', *tip_velocities_tups)

# force
FORCE_VALUE_RANGE = 5
def force_membership_func_neg(x):
    return min(max(-x, 0) / FORCE_VALUE_RANGE, 1)
def force_membership_func_zer(x):
    return max(-abs(x) / FORCE_VALUE_RANGE + 1, 0)
def force_membership_func_pos(x):
    return min(max(x, 0) / FORCE_VALUE_RANGE, 1)

force_domain = np.linspace(-FORCE_DOMAIN, FORCE_DOMAIN, 101)
force_tups = Memebership_display_tuples(
    force_domain,
    force_membership_func_neg,
    force_membership_func_zer,
    force_membership_func_pos,
    )
Display_membership_functions('Force', *force_tups)

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
    u_cart_position_neg = cart_position_func_neg(cart_position)
    u_cart_position_zer = cart_position_func_zer(cart_position)
    u_cart_position_pos = cart_position_func_pos(cart_position)
    
    u_cart_velocity_neg = cart_velocity_func_neg(cart_velocity)
    u_cart_velocity_zer = cart_velocity_func_zer(cart_velocity)
    u_cart_velocity_pos = cart_velocity_func_pos(cart_velocity)
    
    u_pole_angle_neg = pole_angle_membership_func_neg(pole_angle)
    u_pole_angle_zer = pole_angle_membership_func_zer(pole_angle)
    u_pole_angle_pos = pole_angle_membership_func_pos(pole_angle)
    
    u_tip_velocity_neg = tip_velocity_func_neg(tip_velocity)
    u_tip_velocity_zer = tip_velocity_func_zer(tip_velocity)
    u_tip_velocity_pos = tip_velocity_func_pos(tip_velocity)
    
           
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
       kąt zero I pr_t zero wyżej
       (R13)JEŻELI kąt zero I pr_t dodatnia TO siła ujemna
       
       (R14)JEŻELI kąt dodatni I pr_t zero TO siła dodatnia
       
       (R15)JEŻELI kąt dodatni I pr_t dodatnia I pr_c zero TO siła dodatnia
       (R16)JEŻELI kąt dodatni I pr_t dodatnia I pr_c dodatnia TO siła dodatnia
    """
    

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


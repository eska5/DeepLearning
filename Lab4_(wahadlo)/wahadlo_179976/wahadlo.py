import numpy as np
from bisect import bisect

def wah_glob():
    Fmax = 1200
    krokcalk = 0.05
    g = 9.8135
    tar = 0.02
    masawoz = 20
    masawah = 20
    drw = 25
    return Fmax, krokcalk, g, tar, masawoz, masawah, drw

### PARAMETRY ###
EPSILON=0.3

num_of_F = 10
F_max = 1200
available_actions = np.linspace(-F_max, F_max, num_of_F)
gamma = 0.9

cube_dim = 25
Q = np.zeros((cube_dim*cube_dim, num_of_F))

def create_hiperdim(cube_dim):
    cube_deg = np.linspace(-np.pi/2, np.pi/2, cube_dim)
    cube_vel = np.linspace(-np.pi/2, np.pi/2, cube_dim)
    weights = np.zeros(cube_dim*cube_dim)

    return cube_deg, cube_vel, weights

cube_deg, cube_vel, weights = create_hiperdim(cube_dim)

def specify_stimulated(state):
    val_x = bisect(cube_deg, state[0], hi=cube_dim-1)
    val_y = bisect(cube_vel, state[1], hi=cube_dim-1)
    return [val_x * cube_dim + val_y]

def usefullness(Q, wagi, akcja):
    return sum(Q[waga, akcja] for waga in wagi)

def epsilon_greedy(Q, state, epsilon):
    wagi_do_treningu = specify_stimulated(state)

    if np.random.rand() < epsilon:
        losowa_akcja = np.random.randint(0, num_of_F)
        return losowa_akcja, wagi_do_treningu

    najlepsza_akcja = np.argmax([usefullness(Q, wagi_do_treningu, akcja) for akcja in range(num_of_F)])
    return najlepsza_akcja, wagi_do_treningu

def F_FIDX_WAGES(Q, stan, epsilon=EPSILON ,train=False):
    indeks_akcji, wagi = epsilon_greedy(Q, stan, epsilon)
    akcja = available_actions[indeks_akcji]

    if train:
        return akcja, indeks_akcji, wagi

    return akcja

def wahadlo(stan,F):
    Fmax, krokcalk, g, tar, masawoz, masawah, drw = wah_glob()

    if F>Fmax:
        F=Fmax
    if F<-Fmax:
        F=-Fmax

    hh = krokcalk * 0.5
    momwoz = masawoz * drw
    momwah = masawah * drw
    cwoz = masawoz * g
    cwah = masawah * g

    sx=np.sin(stan[0])
    cx=np.cos(stan[0])
    c1=masawoz+masawah*sx*sx
    c2=momwah*stan[1]*stan[1]*sx
    c3=tar*stan[3]*cx

    stanpoch = np.zeros(stan.size)

    stanpoch[0]=stan[1]
    stanpoch[1]=((cwah+cwoz)*sx-c2*cx+c3-F*cx)/(drw*c1)
    stanpoch[2]=stan[3]
    stanpoch[3]=(c2-cwah*sx*cx-c3+F)/c1
    stanh = np.zeros(stan.size)
    for i in range(4):
        stanh[i]=stan[i]+stanpoch[i]*hh
  
    sx=np.sin(stanh[0])
    cx=np.cos(stanh[0])
    c1=masawoz+masawah*sx*sx
    c2=momwah*stanh[1]*stanh[1]*sx
    c3=tar*stanh[3]*cx

    stanpochh = np.zeros(stan.size)
    stanpochh[0]=stanh[1]
    stanpochh[1]=((cwah+cwoz)*sx-c2*cx+c3-F*cx)/(drw*c1)
    stanpochh[2]=stanh[3]
    stanpochh[3]=(c2-cwah*sx*cx-c3+F)/c1
    stann = np.zeros(stan.size)
    for i in range(4):
        stann[i]=stan[i]+stanpochh[i]*krokcalk
    if stann[0] > np.pi:
        stann[0]=stann[0]-2*np.pi
    if stann[0] < -np.pi:
        stann[0]=stann[0]+2*np.pi

    return stann

def nagroda(nowystan):
    kara_za_odchylenie = nowystan[0]**2 +  0.25*nowystan[1]**2 + 0.0025* nowystan[2]**2 + 0.0025* nowystan[3]**2
    kara_za_przewrocenie = (abs(nowystan[0]) >= np.pi / 2) * 1000
    return -kara_za_przewrocenie


def wahadlo_test(stanp):
    Fmax, krokcalk, g, tar, masawoz, masawah, drw = wah_glob()
    pli = open('historia.txt', 'w')
    pli.write("Fmax = " + str(Fmax) + "\n")
    pli.write("krokcalk = " + str(krokcalk) + "\n")
    pli.write("g = " + str(g) + "\n")
    pli.write("tar = " + str(tar) + "\n")
    pli.write("masawoz = " + str(masawoz) + "\n")
    pli.write("masawah = " + str(masawah) + "\n")
    pli.write("drw = " + str(drw) + "\n")

    sr_suma_nagrod = 0
    liczba_krokow = 0
    liczba_stanow_poczatkowych, lparam = stanp.shape
    is_solved = True
    for epizod in range(liczba_stanow_poczatkowych):
        nr_stanup = epizod
        stan = stanp[nr_stanup, :]

        krok = 0
        suma_nagrod_epizodu = 0
        czy_przewrocenie_wahadla = 0
        while (krok < 1000) & (czy_przewrocenie_wahadla == 0):
            krok = krok + 1

            F = F_FIDX_WAGES(Q, stan)

            # wyznaczenie nowego stanu:
            nowystan = wahadlo(stan, F)

            czy_przewrocenie_wahadla = (abs(nowystan[0]) >= np.pi / 2)
            R = nagroda(nowystan)
            suma_nagrod_epizodu = suma_nagrod_epizodu + R

            pli.write(str(epizod + 1) + "  " + str(stan[0]) + "  " + str(stan[1]) + "  " + str(stan[2]) + "  " + str(stan[3]) + "  " + str(F) + "\n")

            stan = nowystan

        sr_suma_nagrod = sr_suma_nagrod + suma_nagrod_epizodu / liczba_stanow_poczatkowych
        liczba_krokow = liczba_krokow + krok
        print("w %d epizodzie suma nagrod = %g, liczba krokow = %d" %(epizod, suma_nagrod_epizodu, krok))

    print("srednia suma nagrod w epizodzie = %g" % (sr_suma_nagrod))
    print("srednia liczba krokow ustania wahadla = %g" % (liczba_krokow/liczba_stanow_poczatkowych))


    pli.close()

    is_solved = is_solved and krok == 1000
    return is_solved

def wahadlo_uczenie():
    liczba_epizodow = 100000000
    alfa = 0.001            # wsp.szybkosci uczenia(moze byc funkcja czasu)
    epsylon = 0.1           # wsp.eksploracji(moze byc funkcja czasu)

    stanp = np.array([[np.pi/6,0, 0, 0],[0, np.pi/3, 0, 0], [0, 0, -10, 1], [0, 0, 0, -10], [np.pi/12, np.pi/6, 0, 0],
                      [np.pi/12, -np.pi/6, 0, 0], [-np.pi/12, np.pi/6, 0, 0], [-np.pi/12, -np.pi/6, 0, 0],
                      [np.pi/12, 0, 0, 0], [0, 0, -10, 10]],dtype=float)
    
    liczba_stanow_poczatkowych, lparam = stanp.shape

    for epizod in range(liczba_epizodow):
        nr_stanup = epizod %  liczba_stanow_poczatkowych
        stan = stanp[nr_stanup, :]

        krok = 0
        czy_wahadlo_przewrocilo_sie = 0
        while (krok < 1000) & (czy_wahadlo_przewrocilo_sie == 0):
            krok = krok + 1

            F, FIDX, WAGES = F_FIDX_WAGES(Q, stan, train=True)

            # wyznaczenie nowego stanu:
            nowystan = wahadlo(stan, F)

            # Calculate possible future states
            possible_states = [wahadlo(nowystan, action) for action in available_actions]
            possible_states_wages = [specify_stimulated(new_state) for new_state in possible_states]
            # Find the maximum utility among the possible future states
            max_usefullness = max(
                sum(Q[waga, akcja] for waga in wagi) 
                for akcja, wagi in enumerate(possible_states_wages)
            )

            czy_wahadlo_przewrocilo_sie = (abs(nowystan[0]) >= np.pi / 2)
            R = nagroda(nowystan)

            for w in WAGES:
                Q[w, FIDX] = Q[w, FIDX] + alfa * (R + gamma * max_usefullness - Q[w, FIDX])


            stan = nowystan

        if epizod % 250 == 0:
            if wahadlo_test(stanp):
                for i in range(cube_dim):
                    for j in range(cube_dim):
                        print(Q[i*cube_dim+j])
                print("Program Pomyślnie rozwiązał zadanie !!!")
                return
            
    print("Program nie rozwiązał zadania :(")



wahadlo_uczenie()



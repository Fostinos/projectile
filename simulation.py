import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plot
from typing import List
import math
import json

MAX_POINTS = 100000
# Ces constantes sont mises à jour par la fonction @get_initial_conditions
CONSTANTES = {
    "g": 0.0,    # gravité
    "A": 0.0,    # constante A
    "B": 0.0,    # constante B
    "Q": None    # Q = None => sans frottement
                 # Q = False => frottement lineaire
                 # Q = True => frottement quadratique
}

def read_configuration()->dict:
    try:
        with open("./config.json", "r") as file:
            reading = json.load(file)
            return reading
    except:
        return None
    
def get_initial_conditions(projectile: dict)->list:
    h:float = projectile.get("z_0", 0.0)
    alpha:float = projectile.get("alpha", 0.0)
    speed:float = projectile.get("vitesse_0", 0.0)
    gravity:float = projectile.get("gravite", 9.8)
    # Mis à jour des constantes utilisées dans la fonction @systeme
    CONSTANTES["g"] = gravity
    CONSTANTES["A"] = 0.0
    CONSTANTES["B"] = 0.0
    frottement = projectile.get("frottement", "SANS")
    surface:float = projectile.get("surface", 0.01)
    mass:float = projectile.get("masse", 1.0)
    rho:float = projectile.get("rho", 0.1)
    Cd:float = projectile.get("Cd", 0.1)
    Cp:float = projectile.get("Cp", 0.1)
    if frottement == "LINEAIRE":
        CONSTANTES["Q"] = False # frottement lineaire
        CONSTANTES["A"] = (rho * surface * (Cd*math.cos(alpha)))/(2 * mass)
        CONSTANTES["B"] = (rho * surface * (Cd*math.sin(alpha)))/(2 * mass)
    elif frottement == "QUADRATIQUE":
        CONSTANTES["Q"] = True # frottement quadratique
        CONSTANTES["A"] = (rho * surface * (Cd*math.cos(alpha) - Cp*math.sin(alpha)))/(2 * mass)
        CONSTANTES["B"] = (rho * surface * (Cd*math.sin(alpha) + Cp*math.cos(alpha)))/(2 * mass)
    else:
        CONSTANTES["Q"] = None # sans frottement
    # Conditions initiales
    x0 = 0.0
    z0 = h
    # Vitesse initiale suivant l'axe X
    u0 = speed * math.sin(alpha)
    # Vitesse initiale suivant l'axe Z
    v0 = speed * math.cos(alpha)
    sys = [x0, z0, u0, v0]
    return sys

def get_movement_duration(projectile: dict)->tuple:
    h:float = projectile.get("z_0", 0.0)
    alpha:float = projectile.get("alpha", 0.0)
    speed:float = projectile.get("vitesse_0", 0.0)
    gravity:float = projectile.get("gravite", 9.8)
    u0 = speed * math.sin(alpha)
    # Intervalle de temps pour le mouvement sans frottement
    t_max = (u0 + math.sqrt( math.pow(u0,2) + 2*gravity*h )) / gravity
    t_span = (0, 10*t_max)
    return t_span

# Définition de la fonction d'événement pour z = -0.5
def hit_below_threshold(t, sys):
    x, z, u, v = sys
    return z + 0.5
hit_below_threshold.terminal = True
hit_below_threshold.direction = -1


# Définition du système d'équations différentielles
def system(t, sys):
    """
        @param t : time
        @param sys : [x, z, u, v]
            x : la position x instatannée
            z : la position z instatannée
            u : la vitesse instatannée suivant l'axe X
            v : la vitesse instatannée suivant l'axe Z
    """
    x, z, u, v = sys
    A, B, G, Q = CONSTANTES["A"], CONSTANTES["B"], CONSTANTES["g"], CONSTANTES["Q"]
    if Q is None: # sans frottement
        du_dt = 0
        dv_dt = -G
    elif Q is False: # frottement lineaire
        du_dt = -A * (u + v)
        dv_dt = -B * (u + v) - G
    elif Q is True: # frottement quadratique
        du_dt = -A * (u**2 + v**2)
        dv_dt = -B * (u**2 + v**2) - G
    dx_dt = u
    dz_dt = v
    return [dx_dt, dz_dt, du_dt, dv_dt]

def plot_intersections(x_points, z_points, color):
    intersections = []
    for i in range(len(z_points)-1):
        if z_points[i] * z_points[i+1] < 0:  # Check if the trajectory crosses z = 0
            # Linear interpolation to find the approximate x-coordinate where z = 0
            x_intersect = x_points[i] - z_points[i] * (x_points[i+1] - x_points[i]) / (z_points[i+1] - z_points[i])
            intersections.append(x_intersect)
    plot.scatter(intersections, [0]*len(intersections), color=color, marker='o')
    for x, y in zip(intersections, [0]*len(intersections)):
        plot.annotate(f'{x:.2f}', (x, 0), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=7)



def main():

    config = read_configuration()
    if config:
        title = config.get("titre", "Projectile")
        gravity = config.get("gravite", 9.8)
        projectiles : List[dict] = config.get("projectiles", [])
        plot.figure("Simulation")
        for projectile in projectiles:
            projectile["gravite"] = gravity
            projectile["alpha"] = math.radians(projectile.get("alpha", 45.0))
            projectile["coleur"] = projectile.get("couleur", "b")
            projectile["label"] = projectile.get("label", "")
            # Vecteur des conditions initiales
            sys0 = get_initial_conditions(projectile)
            # Intervalle de temps pour la solution
            t_span = get_movement_duration(projectile)
            t_eval = np.linspace(t_span[0], t_span[1], MAX_POINTS)
            # Résolution du système d'équations différentielles
            sol = solve_ivp(system, t_span, sys0, t_eval=t_eval, events=hit_below_threshold)
            projectile["x_points"] = sol.y[0]
            projectile["z_points"] = sol.y[1]
            plot.plot(projectile["x_points"], projectile["z_points"], color= projectile["couleur"], label= projectile["label"], linestyle='-')
            plot_intersections(projectile["x_points"], projectile["z_points"], color= projectile["coleur"])
        plot.axhline(0, color='black', linestyle='-')
        plot.xlabel('X-axis')
        plot.ylabel('Y-axis')
        plot.title(title)
        plot.legend()
        plot.grid(True)
        plot.show()

if __name__ == "__main__":
    main()

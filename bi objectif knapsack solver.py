import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_option_menu import option_menu
import time

def knapsack_solver(poids_maximum, poids_object, z1_vect, z2_vect, epsilon):
    start = time.time()
    decision = cp.Variable(len(poids_object), boolean=True)
    contrainte_poid = poids_object @ decision <= poids_maximum
    Z1 = z1_vect @ decision
    p1 = cp.Problem(cp.Maximize(Z1), [contrainte_poid])
    p1.solve(solver=cp.GLPK_MI)

    Z2 = z2_vect @ decision
    p2 = cp.Problem(cp.Maximize(Z2), [contrainte_poid, Z1 == Z1.value])
    p2.solve(solver=cp.GLPK_MI)

    solutions_pareto = [decision.value]
    j = 1
    valeurs_solution_Z1 = [Z1.value]
    valeurs_solution_Z2 = [Z2.value]
    solutions_Z = [(Z1.value, Z2.value)]

    while True:
        p_epsilon = cp.Problem(cp.Maximize(Z1), [contrainte_poid, Z2 >= valeurs_solution_Z2[j - 1] + epsilon])
        p_epsilon.solve(solver=cp.GLPK_MI)

        if (Z1.value, Z2.value) == (None, None):
            break
        else:
            solutions_pareto.append(decision.value)
            valeurs_solution_Z1.append(Z1.value)
            valeurs_solution_Z2.append(Z2.value)
            solutions_Z.append((Z1.value, Z2.value))
            j += 1
    end = time.time()

    st.write("### L'ensemble des solutions efficaces:")
    indexes = [f'X({i+1})' for i in range(len(solutions_pareto))]
    df = pd.DataFrame({
        '': indexes,
        'Solution:': solutions_pareto,
        'Z1': valeurs_solution_Z1,
        'Z2': valeurs_solution_Z2
    })
    table = st.expander("Solutions")
    with table:
        st.table(df.set_index(df.columns[0]))

    st.write("### Front de pareto:")
    plt.figure(figsize=(8, 6))
    plt.plot(valeurs_solution_Z1, valeurs_solution_Z2, marker='o', linestyle='-')
    plt.xlabel('Z1')
    plt.ylabel('Z2')
    plt.title('Front de pareto')
    st.pyplot(plt)

    st.write(f'### Temps de calcul: {round((end - start), 4)} seconds')


def main():
    set_custom_style()

    selected = option_menu(
        menu_title="",
        options=["Manual", "Random"],
        default_index=0,
        orientation='horizontal'
    )

    if selected == "Manual":
        text = "Développé par: HAMDANE Ayoub et HAMMACHE Ghiles"
        st.markdown(f"<p style='font-weight: bold;'>{text}</p>", unsafe_allow_html=True)
        st.title('Bi-objective Knapsack Problem Solver with the epsilon constraint method')
        poids_maximum = st.number_input('Poids maximum du sac:')
        n = st.number_input("Nombre d'objets:", step=1)
        epsilon = st.number_input("Valeur d'epsilon:")

        poids_object = np.array([])
        z1_vect = np.array([])
        z2_vect = np.array([])

        solve = st.button('Résoudre')
        st.write("### L'ensemble des objets:")
        with st.expander('Objets:'):
            for i in range(int(n)):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.write(f'## Objet {i + 1}')
                with col2:
                    poid = st.number_input(f"Poids de l'objet:", key=f"poids_{i+1}")
                with col3:
                    valeur1 = st.number_input(f"Valeur 1 de l'objet:", key=f"z1_{i+1}")
                with col4:
                    valeur2 = st.number_input(f"Valeur 2 de l'objet:", key=f"z2_{i+1}")

                poids_object = np.append(poids_object, poid)
                z1_vect = np.append(z1_vect, valeur1)
                z2_vect = np.append(z2_vect, valeur2)

        if solve:
            knapsack_solver(poids_maximum, poids_object, z1_vect, z2_vect, epsilon)

    if selected == "Random":
        text = "Développé par: HAMDANE Ayoub et HAMMACHE Ghiles"
        st.markdown(f"<p style='font-weight: bold;'>{text}</p>", unsafe_allow_html=True)
        st.title('Bi-objective Knapsack Problem Solver with the epsilon constraint method')
        poids_maximum = st.number_input('Poids maximum du sac:')
        n = st.number_input("Nombre d'objets:", step=1)
        epsilon = st.number_input("Valeur d'epsilon:")

        poids_object = np.array([])
        z1_vect = np.array([])
        z2_vect = np.array([])

        random = st.button('Randomize')
        if random:
            st.write("### L'ensemble des objets:")
            with st.expander('Objets:'):
                for i in range(int(n)):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.write(f'### Objet {i + 1}')
                    with col2:
                        poid = st.number_input(f"Poids de l'objet:", key=f"Rpoids_{i+1}", value=np.random.randint(0, poids_maximum) + np.random.random())
                    with col3:
                        valeur1 = st.number_input(f"Valeur 1 de l'objet:", key=f"Rz1_{i+1}", value=abs(np.random.randint(0, 100) + np.random.random()))
                    with col4:
                        valeur2 = st.number_input(f"Valeur 2 de l'objet:", key=f"Rz2_{i+1}", value=abs(np.random.randint(0, 100) + np.random.random()))

                    poids_object = np.append(poids_object, poid)
                    z1_vect = np.append(z1_vect, valeur1)
                    z2_vect = np.append(z2_vect, valeur2)

            knapsack_solver(poids_maximum, poids_object, z1_vect, z2_vect, epsilon)

def set_custom_style():
    st.markdown("""
        <style>
        body {
            background-color: #f0f2f6;
        }
        .stApp {
            background-color: #f9f9ff;
        }
        h1, h2, h3, h4 {
            color: #003366;
        }
        .css-1d391kg {
            color: #003366 !important;
        }
        .stButton>button {
            background-color: #3366cc;
            color: white;
            border: None;
            border-radius: 8px;
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background-color: #254b9b;
        }
        .stNumberInput>div>div>input {
            background-color: #ffffff;
        }
        .stTable {
            background-color: #ffffff;
        }
        </style>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()

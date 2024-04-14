# Importamos las librerías necesarias
from flask import Flask, render_template, request, send_from_directory
import sympy as sp
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import os

# Creamos una instancia de la clase Flask
app = Flask(__name__)

#Rutas de la página
@app.route('/')
def home():
    # Renderizamos la plantilla del menu
    return render_template('menu.html')

@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/S_Dorada.html')
def S_Dorada():
    return render_template('S_Dorada.html')

@app.route('/menu.html')
def menu():
    return render_template('menu.html')






#Método de la sección dorada
def seccion_dorada(fun_str, li, ls, maxite, exac):
    def graficos():
        # Definimos una función interna para evaluar la función en un punto dado
        def f(x_val):
            return fun.subs(x, x_val)
        #Configuramos los colores y estilos de los gráficos
        plt.rcParams['figure.facecolor'] = '#2d2d2d'  # fondo oscuro
        plt.rcParams['text.color'] = '#eaeaea'  # texto en gris claro
        plt.rcParams['axes.facecolor'] = '#2d2d2d'  # fondo de los ejes oscuro
        plt.rcParams['axes.edgecolor'] = '#eaeaea'  # bordes de los ejes en gris claro
        plt.rcParams['axes.labelcolor'] = '#eaeaea'  # etiquetas de los ejes en gris claro
        plt.rcParams['xtick.color'] = '#eaeaea'  # ticks del eje x en gris claro
        plt.rcParams['ytick.color'] = '#eaeaea' # ticks del eje y en gris claro

        fig, ax = plt.subplots()
        ax.set_xticks(range(lir-1, lsr+1))
        ax.set_yticks(range(0, 20, 5))
        ax.grid(True, color='#4a4a4a')    
        fig.patch.set_facecolor('#2d2d2d')
        ax.patch.set_facecolor('#2d2d2d')
        #Generamos los valores de X y Y para la gráfica
        x_vals=np.linspace(li, ls, 100)
        y_vals=[f(x_val) for x_val in x_vals]
        #Dibujamos la gráfica
        plt.plot(x_vals, y_vals, color='#e67300')
        plt.axhline(0, color='#eaeaea', linewidth=.5)  # línea del eje x en gris claro
        plt.axvline(0, color='#eaeaea', linewidth=.5) 
        # Dibujamos las líneas verticales y los puntos en los márgenes
        for i in range (len(margenes)):
            xpun=margenes[i]
            ypun=fun.subs(x, margenes[i])
            punto=(xpun, ypun)
            plt.vlines(xpun, ymin=0, ymax=ypun, linestyle='-', color='red', alpha=0.5)
            plt.plot(punto[0], punto[1], marker='o')
            if (i==len(margenes)-1 or i==len(margenes)-2):
                plt.vlines(xpun, ymin=0, ymax=ypun, linestyle='-', color='yellow')
                plt.plot(punto[0], punto[1], marker='o')
        # Configuramos las etiquetas y el título de la gráfica       
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Raíz en: " + str(matriz[4]), fontsize=20)
        plt.suptitle(str(fun), fontsize=14)
        # Guardamos la gráfica en un archivo de mi carpeta estática
        plt.savefig('static/plot.png')
    #Convertimos la cadena de la función en una expresión simbólica
    x=sp.Symbol('x')
    fun=sp.sympify(fun_str)
    lir = round(li)-1
    lsr = round(ls)+1
    matriz = [0] * 10
    margenes = []
    fin = 0
    i = 1
    aureo = 0.618
    aureo2 = aureo**2
    matriz[1] = li
    matriz[2] = ls
    iteraciones = []
    #Iteramos hasta que se cumpla la condición de parada de la sección dorada
    while (fin != 1) and (matriz[0] < maxite):
        if matriz[0] == 0:
            margenes.append(matriz[1])
            margenes.append(matriz[2])
        matriz[0] += 1
        resta = matriz[2] - matriz[1]
        matriz[3] = resta * (aureo)
        matriz[4] = resta * (aureo2)
        matriz[5] = matriz[1] + matriz[3]
        matriz[6] = matriz[1] + matriz[4]
        matriz[7] = fun.subs(x, matriz [5])
        matriz[8] = fun.subs(x, matriz [6])
        matriz[9] = abs(matriz[6] - matriz[5])
        matriz = [round(numero, 10) for numero in matriz]
        iteraciones.append({
            'iteracion': matriz[0],
            'limites': [matriz[1], matriz[2]],
            'a': matriz[3],
            'b': matriz[4],
            'x1': matriz[5],
            'fx1': matriz[7],
            'x2': matriz[6],
            'fx2': matriz[8],
            'margen': matriz[9]
        })
        # Define los colores
        background_color = '#2d2d2d'
        text_color = '#eaeaea'
        line_color = '#e67300'
        grid_color = '#4a4a4a'
        if matriz[9] <= exac:
            fin=1
        else:
            if matriz[8] > matriz[7]:
                matriz[1] = matriz[1]
                matriz[2] = matriz[5]
                margenes.append(matriz[2])
            else:
                matriz[1] = matriz[6]
                matriz[2] = matriz[2]
                margenes.append(matriz[1])
    graficos()
    #Retornamos o devolvemos los resultados
    return {
        'raiz': [matriz[1], matriz[2]],
        'margen': matriz[9],
        'exactitud': exac,
        'iteraciones': maxite,
        'grafica': 'static/plot.png',
        'detalles': iteraciones,
        'colors': {
            'background': background_color,
            'text': text_color,
            'line': line_color,
            'grid': grid_color
        }
    }

@app.route('/S_Dorada', methods=['GET', 'POST'])
def s_dorada():
    #Definimos la ruta para el método de la sección dorada
    if request.method == 'POST':
        # Si el método es POST, obtenemos los datos del formulario y calculamos los resultados
        fun_str = request.form['func']
        li = float(request.form['li'])
        ls = float(request.form['ls'])
        maxite = int(request.form['maxite'])
        exac = float(request.form['exac'])
        results = seccion_dorada(fun_str, li, ls, maxite, exac)
        # Renderizamos la plantilla de resultados
        return render_template('resultsD.html', results=results)
    #si el metodo no es POST, se renderiza la plantilla de la sección dorada
    return render_template('S_Dorada.html')

#------------------------------------------------------------------------------------------------------------
# Definimos la ruta para el cálculo del gradiente
@app.route('/calculate', methods=['POST'])
def calculate():
    # Obtenemos los datos del formulario
    func = request.form['func']
    learning_rate = float(request.form['learning_rate'])
    epsilon = float(request.form['epsilon'])

    # Definimos la función y su derivada
    x = sp.symbols('x')
    f = sp.sympify(func)
    df = sp.diff(f, x)

    # Calculamos el valor máximo de la función
    x_max = sp.solve(df, x)[0]
    max_value = f.subs(x, x_max)

    # Inicializamos las variables para el método del gradiente
    x = 0
    value = f.subs(sp.symbols('x'), x)

    # Creamos un DataFrame para almacenar los resultados
    results = pd.DataFrame(columns=['Iteración', 'x', 'Valor'])
    iteration = 0

    # Ejecutamos el método del gradiente
    while abs(value - max_value) > epsilon and iteration < 1000:
        x = x + learning_rate * df.subs(sp.symbols('x'), x)
        value = f.subs(sp.symbols('x'), x)
        results.loc[iteration] = [iteration, float(x), float(value)]
        iteration += 1

    # Creamos una gráfica con los resultados
    x_vals = np.linspace(-10, 10, 400)
    y_vals = np.array([float(f.subs(sp.symbols('x'), val).evalf()) for val in x_vals])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name=str(f)))
    fig.add_trace(go.Scatter(x=[float(x)], y=[float(value)], mode='markers', name='Valor máximo aproximado'))  # plot the maximum point
    fig.add_trace(go.Scatter(x=[float(x_max)], y=[float(max_value)], mode='markers', name='Valor máximo'))  # plot the actual maximum point

    # Guardamos la gráfica como un archivo HTML
    if not os.path.exists('templates'):
        os.makedirs('templates')
    fig.write_html('templates/plot.html')

    # Renderizamos la plantilla de los resultados
    return render_template('results.html', value=float(value), x=float(x), results=results.to_html(), plot='plot.html')

@app.route('/templates/<path:path>')
def send_plot(path):
    # Definimos la ruta para enviar el gráfico al usuario
    return send_from_directory('templates', path)

#ejecutamos la aplicación
if __name__ == "__main__":
    app.run(debug=True)
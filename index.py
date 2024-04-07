# Importamos las bibliotecas necesarias
from flask import Flask, render_template, request
import sympy as sp
import pandas as pd

# Creamos una instancia de la clase Flask
app = Flask(__name__)

# Definimos la ruta para la página de inicio
@app.route('/')
def home():
    # Renderizamos la plantilla index.html
    return render_template('index.html')

# Definimos la ruta para calcular los resultados
@app.route('/calculate', methods=['POST'])
def calculate():
    # Obtenemos los datos del formulario
    func = request.form['func']
    learning_rate = float(request.form['learning_rate'])
    epsilon = float(request.form['epsilon'])

    # Definimos x como un símbolo
    x = sp.symbols('x')
    # Convertimos la función ingresada por el usuario en una función sympy
    f = sp.sympify(func)

    # Calculamos la derivada de la función
    df = sp.diff(f, x)
    # Resolvemos la ecuación df = 0 para encontrar el máximo
    x_max = sp.solve(df, x)[0]
    # Evaluamos la función en el máximo para obtener el valor máximo
    max_value = f.subs(x, x_max)

    # Inicializamos x a 0
    x = 0
    # Evaluamos la función en x
    value = f.subs(sp.symbols('x'), x)

    # Creamos un DataFrame para almacenar los resultados de cada iteración
    results = pd.DataFrame(columns=['Iteración', 'x', 'Valor'])
    iteration = 0
    # Iteramos hasta que la diferencia entre el valor actual y el máximo sea menor que epsilon
    while abs(value - max_value) > epsilon and iteration < 1000:
        # Actualizamos x usando el método del gradiente ascendente
        x = x + learning_rate * df.subs(sp.symbols('x'), x)
        # Evaluamos la función en el nuevo x
        value = f.subs(sp.symbols('x'), x)
        # Añadimos los resultados de esta iteración al DataFrame
        results.loc[iteration] = [iteration, x, value]
        iteration += 1

    # Renderizamos la plantilla results.html con los resultados
    return render_template('results.html', value=value, x=x, results=results.to_html())

# Ejecutamos la aplicación
if __name__ == "__main__":
    app.run(debug=True)
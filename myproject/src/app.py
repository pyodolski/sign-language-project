from flask import Flask, render_template

app = Flask(__name__, template_folder='templates')

@app.route('/')
def mode_selector():
    return render_template('index.html')

@app.route('/ksl')
def ksl_mode():
    return render_template('ksl.html')


@app.route('/asl')
def asl_mode():
    return "<h2>ASL 인식 페이지 (준비 중)</h2>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002,debug=True)

import Home from './pages/Home'

function App() {
    return (
        <div className="app">
            <header className="header">
                <div className="header__brand">
                    <img src="/ghost.svg" alt="Polyglot Ghost" className="header__icon" />
                    <span className="header__title">Polyglot Ghost</span>
                </div>
                <span className="header__badge">v1.0 — Acoustic AI</span>
            </header>

            <main className="main">
                <Home />
            </main>

            <footer className="footer">
                Polyglot Ghost &copy; {new Date().getFullYear()} — Voice Deepfake Detection
            </footer>
        </div>
    )
}

export default App

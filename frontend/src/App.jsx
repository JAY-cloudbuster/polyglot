import { useState } from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import CinematicIntro from './components/CinematicIntro'
import Navbar from './components/Navbar'
import RhythmBackground from './components/RhythmBackground'
import Landing from './pages/Landing'
import RecordPage from './pages/RecordPage'
import UploadPage from './pages/UploadPage'

function App() {
    const [introComplete, setIntroComplete] = useState(false)

    return (
        <BrowserRouter>
            {/* Cinematic intro overlay — unmounts itself after completion */}
            {!introComplete && (
                <CinematicIntro onComplete={() => setIntroComplete(true)} />
            )}

            <RhythmBackground />
            <div style={{ position: 'relative', zIndex: 1 }}>
                <Navbar />
                <Routes>
                    <Route path="/" element={<Landing />} />
                    <Route path="/record" element={<RecordPage />} />
                    <Route path="/upload" element={<UploadPage />} />
                </Routes>
                <footer className="footer">
                    Polyglot Ghost &copy; {new Date().getFullYear()}
                </footer>
            </div>
        </BrowserRouter>
    )
}

export default App

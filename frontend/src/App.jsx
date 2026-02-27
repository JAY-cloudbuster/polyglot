import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import RhythmBackground from './components/RhythmBackground'
import Landing from './pages/Landing'
import RecordPage from './pages/RecordPage'
import UploadPage from './pages/UploadPage'

function App() {
    return (
        <BrowserRouter>
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

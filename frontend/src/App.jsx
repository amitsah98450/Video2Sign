import { useState } from "react";
import Header from "./components/Header";
import VideoUpload from "./components/VideoUpload";
import TranscriptionView from "./components/TranscriptionView";
import SignLanguagePlayer from "./components/SignLanguagePlayer";
import "./App.css";

function App() {
  const [currentView, setCurrentView] = useState("upload"); // upload | transcription | sign
  const [transcription, setTranscription] = useState(null);
  const [signData, setSignData] = useState(null);

  const handleTranscriptionComplete = (data) => {
    setTranscription(data.transcription);
    setCurrentView("transcription");
  };

  const handleConvertToSign = (signs) => {
    setSignData(signs);
    setCurrentView("sign");
  };

  const handleReset = () => {
    setCurrentView("upload");
    setTranscription(null);
    setSignData(null);
  };

  const handleBackToTranscription = () => {
    setCurrentView("transcription");
    setSignData(null);
  };

  return (
    <div className="app">
      <div className="bg-glow bg-glow-1" />
      <div className="bg-glow bg-glow-2" />
      <div className="bg-glow bg-glow-3" />

      <Header
        currentView={currentView}
        onReset={handleReset}
        onBack={currentView === "sign" ? handleBackToTranscription : null}
      />

      <main className="main-content">
        <div className="container">
          {currentView === "upload" && (
            <VideoUpload onComplete={handleTranscriptionComplete} />
          )}
          {currentView === "transcription" && transcription && (
            <TranscriptionView
              transcription={transcription}
              onConvertToSign={handleConvertToSign}
            />
          )}
          {currentView === "sign" && signData && (
            <SignLanguagePlayer
              signData={signData}
              onBack={handleBackToTranscription}
            />
          )}
        </div>
      </main>
    </div>
  );
}

export default App;

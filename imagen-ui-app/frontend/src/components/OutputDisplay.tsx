import './OutputDisplay.css'

interface OutputDisplayProps {
  outputImage: string
  prompt: string
  onReset: () => void
  onFeedback: () => void
}

function OutputDisplay({ outputImage, prompt, onReset, onFeedback }: OutputDisplayProps) {
  const handleDownload = () => {
    const link = document.createElement('a')
    link.href = outputImage
    link.download = `generated-image-${Date.now()}.png`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  return (
    <div className="output-display">
      <div className="output-header">
        <h2>Generated Image</h2>
        <button onClick={onReset} className="btn-close">
          New Generation
        </button>
      </div>

      <div className="output-content">
        <div className="image-container">
          <img src={outputImage} alt="Generated output" className="output-image" />
        </div>

        <div className="prompt-display">
          <h3>Prompt Used:</h3>
          <p>{prompt}</p>
        </div>
      </div>

      <div className="output-actions">
        <button onClick={handleDownload} className="btn btn-download">
          <svg
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="7 10 12 15 17 10" />
            <line x1="12" y1="15" x2="12" y2="3" />
          </svg>
          Download
        </button>

        <button onClick={onFeedback} className="btn btn-feedback">
          <svg
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z" />
          </svg>
          Provide Feedback
        </button>
      </div>
    </div>
  )
}

export default OutputDisplay

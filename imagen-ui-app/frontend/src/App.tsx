import { useState } from 'react'
import axios from 'axios'
import ImageUploader from './components/ImageUploader'
import OutputDisplay from './components/OutputDisplay'
import FeedbackModal from './components/FeedbackModal'
import './App.css'

// In development, proxy is configured in setupProxy.js
// In production, both frontend and backend are served from same origin
const API_URL = process.env.NODE_ENV === 'development' ? '' : ''

function App() {
  const [image1, setImage1] = useState<string | null>(null)
  const [image2, setImage2] = useState<string | null>(null)
  const [prompt, setPrompt] = useState('')
  const [outputImage, setOutputImage] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showFeedbackModal, setShowFeedbackModal] = useState(false)

  const handleImageUpload = (imageNumber: number, file: File) => {
    const reader = new FileReader()
    reader.onloadend = () => {
      const base64String = reader.result as string
      if (imageNumber === 1) {
        setImage1(base64String)
      } else {
        setImage2(base64String)
      }
    }
    reader.readAsDataURL(file)
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!image1 || !image2 || !prompt.trim()) {
      setError('Please upload both images and enter a prompt')
      return
    }

    setIsLoading(true)
    setError(null)
    setOutputImage(null)

    try {
      const response = await axios.post(`${API_URL}/api/predict`, {
        image1: image1,
        image2: image2,
        prompt: prompt,
        num_inference_steps: 40,
        true_cfg_scale: 4.0,
        guidance_scale: 1.0
      })

      setOutputImage(response.data.output_image)
    } catch (err) {
      console.error('Error:', err)
      setError((err as any).response?.data?.detail || 'Failed to generate image. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }

  const handleReset = () => {
    setImage1(null)
    setImage2(null)
    setPrompt('')
    setOutputImage(null)
    setError(null)
  }

  const handleFeedbackSubmit = async (feedbackData: { feedback_type: string; feedback_text: string | null }) => {
    try {
      await axios.post(`${API_URL}/api/feedback`, {
        ...feedbackData,
        prompt: prompt,
        session_id: Date.now().toString()
      })
      setShowFeedbackModal(false)
    } catch (err) {
      console.error('Error submitting feedback:', err)
      alert('Failed to submit feedback. Please try again.')
    }
  }

  return (
    <div className="app">
      <div className="container">
        <header className="header">
          <h1>Image-to-Image Editor</h1>
          <p>Upload two images and describe how you want them combined</p>
        </header>

        <div className="content">
          {!outputImage ? (
            <form onSubmit={handleSubmit} className="input-section">
              <div className="image-uploads">
                <ImageUploader
                  label="Image 1"
                  imagePreview={image1}
                  onImageUpload={(file) => handleImageUpload(1, file)}
                />
                <ImageUploader
                  label="Image 2"
                  imagePreview={image2}
                  onImageUpload={(file) => handleImageUpload(2, file)}
                />
              </div>

              <div className="prompt-section">
                <label htmlFor="prompt">Prompt</label>
                <textarea
                  id="prompt"
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder="Describe how you want to combine the images..."
                  rows={4}
                  disabled={isLoading}
                />
              </div>

              {error && (
                <div className="error-message">
                  {error}
                </div>
              )}

              <div className="button-group">
                <button
                  type="submit"
                  className="btn btn-primary"
                  disabled={isLoading || !image1 || !image2 || !prompt.trim()}
                >
                  {isLoading ? 'Generating...' : 'Generate Image'}
                </button>
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={handleReset}
                  disabled={isLoading}
                >
                  Reset
                </button>
              </div>
            </form>
          ) : (
            <OutputDisplay
              outputImage={outputImage}
              prompt={prompt}
              onReset={handleReset}
              onFeedback={() => setShowFeedbackModal(true)}
            />
          )}
        </div>
      </div>

      {showFeedbackModal && (
        <FeedbackModal
          onClose={() => setShowFeedbackModal(false)}
          onSubmit={handleFeedbackSubmit}
        />
      )}
    </div>
  )
}

export default App

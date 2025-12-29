import { useState } from 'react'
import './FeedbackModal.css'

interface FeedbackModalProps {
  onClose: () => void
  onSubmit: (feedbackData: { feedback_type: string; feedback_text: string | null }) => Promise<void>
}

function FeedbackModal({ onClose, onSubmit }: FeedbackModalProps) {
  const [feedbackType, setFeedbackType] = useState<string | null>(null)
  const [feedbackText, setFeedbackText] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!feedbackType) {
      alert('Please select thumbs up or thumbs down')
      return
    }

    setIsSubmitting(true)

    try {
      await onSubmit({
        feedback_type: feedbackType,
        feedback_text: feedbackText.trim() || null
      })
    } catch (error) {
      console.error('Error submitting feedback:', error)
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>Provide Feedback</h2>
          <button className="modal-close" onClick={onClose}>
            <svg
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        <form onSubmit={handleSubmit} className="modal-body">
          <div className="feedback-rating">
            <p>How would you rate the generated image?</p>
            <div className="rating-buttons">
              <button
                type="button"
                className={`rating-btn ${feedbackType === 'thumbs_up' ? 'active thumbs-up' : ''}`}
                onClick={() => setFeedbackType('thumbs_up')}
              >
                <svg
                  width="32"
                  height="32"
                  viewBox="0 0 24 24"
                  fill={feedbackType === 'thumbs_up' ? 'currentColor' : 'none'}
                  stroke="currentColor"
                  strokeWidth="2"
                >
                  <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3" />
                </svg>
                <span>Thumbs Up</span>
              </button>

              <button
                type="button"
                className={`rating-btn ${feedbackType === 'thumbs_down' ? 'active thumbs-down' : ''}`}
                onClick={() => setFeedbackType('thumbs_down')}
              >
                <svg
                  width="32"
                  height="32"
                  viewBox="0 0 24 24"
                  fill={feedbackType === 'thumbs_down' ? 'currentColor' : 'none'}
                  stroke="currentColor"
                  strokeWidth="2"
                >
                  <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17" />
                </svg>
                <span>Thumbs Down</span>
              </button>
            </div>
          </div>

          <div className="feedback-text">
            <label htmlFor="feedback-input">Additional Comments (Optional)</label>
            <textarea
              id="feedback-input"
              value={feedbackText}
              onChange={(e) => setFeedbackText(e.target.value)}
              placeholder="Tell us more about your experience..."
              rows={4}
            />
          </div>

          <div className="modal-footer">
            <button
              type="button"
              className="btn btn-cancel"
              onClick={onClose}
              disabled={isSubmitting}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="btn btn-submit"
              disabled={!feedbackType || isSubmitting}
            >
              {isSubmitting ? 'Submitting...' : 'Submit Feedback'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

export default FeedbackModal

import { useTheme } from './ThemeContext'
import './ThemeToggle.css'

export function ThemeToggle({ className = "" }) {
  const { isDark, toggleTheme } = useTheme()

  const handleToggle = () => {
    toggleTheme()
  }

  return (
    <div
      className={`theme-toggle ${isDark ? 'dark' : 'light'} ${className}`}
      onClick={handleToggle}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault()
          handleToggle()
        }
      }}
    >
      <div className="toggle-track">
        <div className={`toggle-thumb ${isDark ? 'dark' : 'light'}`}>
          <div className="icon">
            {isDark ? (
              // Moon icon
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5}>
                <path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"></path>
              </svg>
            ) : (
              // Sun icon  
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5}>
                <circle cx="12" cy="12" r="4"></circle>
                <path d="m12 2 0 2"></path>
                <path d="m12 20 0 2"></path>
                <path d="m4.93 4.93 1.41 1.41"></path>
                <path d="m17.66 17.66 1.41 1.41"></path>
                <path d="m2 12 2 0"></path>
                <path d="m20 12 2 0"></path>
                <path d="m6.34 17.66-1.41 1.41"></path>
                <path d="m19.07 4.93-1.41 1.41"></path>
              </svg>
            )}
          </div>
        </div>
        <div className={`secondary-icon ${isDark ? 'dark' : 'light'}`}>
          {isDark ? (
            // Sun icon (inactive)
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5}>
              <circle cx="12" cy="12" r="4"></circle>
              <path d="m12 2 0 2"></path>
              <path d="m12 20 0 2"></path>
              <path d="m4.93 4.93 1.41 1.41"></path>
              <path d="m17.66 17.66 1.41 1.41"></path>
              <path d="m2 12 2 0"></path>
              <path d="m20 12 2 0"></path>
              <path d="m6.34 17.66-1.41 1.41"></path>
              <path d="m19.07 4.93-1.41 1.41"></path>
            </svg>
          ) : (
            // Moon icon (inactive)
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5}>
              <path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"></path>
            </svg>
          )}
        </div>
      </div>
    </div>
  )
}
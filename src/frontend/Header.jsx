import Logo from "./Factify-Logo.jpeg"
import { Link } from "react-router-dom"

export default function Header() {
    return (
     
      
        <div>
        <header className="header">
          <div className="container">
            <div className="logo">
            <Link to="/">
              <img src={Logo} alt="FACTIFY Logo" className="logo-image" />
            </Link>
              
            </div>
            <nav className="nav">
              <ul className="nav-list">
                {/* Remove Features link from here */}
              </ul>
            </nav>
         
            <div className="actions">
              <Link to="/" className="search-button">
                Search
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M21 21L16.65 16.65M19 11C19 15.4183 15.4183 19 11 19C6.58172 19 3 15.4183 3 11C3 6.58172 6.58172 3 11 3C15.4183 3 19 6.58172 19 11Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </Link>
              {/* Add Features link here */}
              <Link to="/features" className="search-button">
                Features
              </Link>
              <Link to="/contact" className="search-button contact-button">
                Contact
              </Link>
            </div>
          </div>
        </header>

        
        
      </div>
    )
  }
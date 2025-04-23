import { useState } from "react";
import Logo from "./Factify-Logo.jpeg"
import { Link } from "react-router-dom"

export default function Header() {
    const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

    const toggleMobileMenu = () => {
        setIsMobileMenuOpen(!isMobileMenuOpen);
    };

    // Function to close menu if open
    const closeMobileMenu = () => {
        if (isMobileMenuOpen) {
            setIsMobileMenuOpen(false);
        }
    };

    return (
        <>
            {isMobileMenuOpen && (
                <div className="mobile-menu-overlay" onClick={toggleMobileMenu}></div>
            )}
            <header className="header">
                <div className="container">
                    <div className="logo">
                        {/* Close menu when logo is clicked on mobile */}
                        <Link to="/" onClick={closeMobileMenu}>
                            <img src={Logo} alt="FACTIFY Logo" className="logo-image" />
                        </Link>
                    </div>

                    {/* Hamburger Button (Mobile Only) */}
                    <button
                        className="hamburger-button"
                        aria-label="Toggle menu"
                        aria-expanded={isMobileMenuOpen}
                        onClick={toggleMobileMenu}
                    >
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M3 12H21" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                            <path d="M3 6H21" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                            <path d="M3 18H21" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        </svg>
                    </button>

                    {/* Mobile Navigation Panel */}
                    <nav className={`nav ${isMobileMenuOpen ? 'mobile-menu-open' : ''}`}>
                        {/* Add a close button inside the mobile menu */}
                        <button className="close-menu-button" onClick={toggleMobileMenu} aria-label="Close menu">
                            &times; {/* Simple 'X' icon */}
                        </button>
                        <ul className="nav-list">
                            {/* Links for mobile menu - Use nav-link-mobile class */}
                            <li><Link to="/" className="nav-link-mobile" onClick={toggleMobileMenu}>Search</Link></li>
                            <li><Link to="/features" className="nav-link-mobile" onClick={toggleMobileMenu}>Features</Link></li>
                            <li><Link to="/contact" className="nav-link-mobile" onClick={toggleMobileMenu}>Contact</Link></li>
                        </ul>
                    </nav>

                    {/* Desktop Actions (Original Buttons) */}
                    <div className="actions desktop-actions">
                        {/* Add nav-link class back */}
                        <Link to="/" className="search-button nav-link">
                            Search
                            {/* Search Icon SVG */}
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M21 21L16.65 16.65M19 11C19 15.4183 15.4183 19 11 19C6.58172 19 3 15.4183 3 11C3 6.58172 6.58172 3 11 3C15.4183 3 19 6.58172 19 11Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>
                        </Link>
                        <Link to="/features" className="search-button nav-link">Features</Link>
                        <Link to="/contact" className="search-button contact-button nav-link">Contact</Link>
                    </div>
                </div>
            </header>
        </>
    )
}
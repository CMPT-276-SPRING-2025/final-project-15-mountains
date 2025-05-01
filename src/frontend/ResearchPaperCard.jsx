import React, { useState } from 'react';
import './styles/ResearchPaperCard.css';

const ResearchPaperCard = ({
  title,
  author,
  date,
  abstract,
  categories = [],
  publisher = '',
  badgeText = '',
  doi = '10.1002/abc.12345',
  published = '28/01/2023',
  source = 'crossref',
  url = '',
  citation_count = 0
}) => {
  const [isExpanded, setIsExpanded] = useState(false);

  // Format date to DD/MM/YYYY if provided as a Date object
  const formattedDate = typeof date === 'object' && date instanceof Date
    ? `${date.getDate().toString().padStart(2, '0')}/${(date.getMonth() + 1).toString().padStart(2, '0')}/${date.getFullYear()}`
    : date;

  // Helper function to get the appropriate CSS class for each source
  const getSourceClass = (sourceString) => {
    const sourceLower = sourceString.toLowerCase();
    
    if (sourceLower.includes('crossref')) return 'source-crossref';
    if (sourceLower.includes('openalex')) return 'source-openalex';
    if (sourceLower.includes('semantic')) return 'source-semantic-scholar';
    if (sourceLower.includes('pubmed')) return 'source-pubmed';
    
    // Default fallback
    return 'source-default';
  };

  // Toggle card expansion
  const toggleExpansion = () => {
    setIsExpanded(!isExpanded);
  };

  return (
    <div className={`research-paper-card ${isExpanded ? 'expanded' : ''}`}>
      {/* Badge */}
      {badgeText && (
        <div className="badge">
          {badgeText}
        </div>
      )}
      
      {/* Title Section - Now Clickable */}
      <h1 className="paper-title">
        {url ? (
          <a href={url} target="_blank" rel="noopener noreferrer" className="title-link">
            {title}
          </a>
        ) : (
          title
        )}
      </h1>
      
      {/* Author and Date */}
      <div className="author-date">
        <span>{author}</span>
        {author && formattedDate && <span> - </span>}
        <span>{formattedDate}</span>
      </div>
      
      {/* Categories */}
      <div className="categories">
        {categories.map((category, index) => (
          <span key={index} className="category">
            {category}
          </span>
        ))}
      </div>
      
      {/* DOI and Publication Info */}
      <div className="doi-info">
        <span>DOI: {doi}</span>
        <span>Published: {published}</span>
        <span className={`source-label ${getSourceClass(source)}`}>
          Source: {source}
        </span>
        <span className="citation-count">Citations: {citation_count}</span>
      </div>
      
      {/* URL/Link to Paper - Keeping the existing button */}
      {url && (
        <div className="paper-url">
          <a href={url} target="_blank" rel="noopener noreferrer">
            Access Full Paper ↗
          </a>
        </div>
      )}
      
      {/* Abstract Section - Now Collapsible via CSS */}
      <div className="abstract">
        {/* Desktop title - hidden on mobile via CSS */}
        <h2 className="abstract-title-desktop">Abstract</h2>

        {/* Abstract content - Always rendered, visibility controlled by CSS */}
        <p className="abstract-content">{abstract}</p>

        {/* Mobile toggle button - hidden on desktop via CSS */}
        <button onClick={toggleExpansion} className="toggle-details-button">
          {isExpanded ? 'Hide Details' : 'Show Details'}
          <span className={`arrow ${isExpanded ? 'up' : 'down'}`}></span>
        </button>
      </div>
      
      {/* Publisher */}
      {publisher && (
        <div className="publisher">
          © {new Date().getFullYear()} {publisher}
        </div>
      )}
    </div>
  );
};

export default ResearchPaperCard;

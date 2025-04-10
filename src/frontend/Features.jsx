import React from 'react';
import './index.css'; // Assuming styles will be added here or imported via App
// If you plan to use icons, import them here
// import { IconName } from 'react-icons/fa'; // Example

const features = [
  {
    // icon: <IconName />, // Replace with actual icon component or element
    icon: "🔬", // Placeholder emoji
    title: "AI-Powered Claim Verification",
    description: "Submit a scientific claim and our AI analyzes relevant research papers to determine its validity.",
  },
  {
    icon: "🔍",
    title: "Evidence Identification",
    description: "Pinpoints specific evidence within research papers that supports or refutes the claim.",
  },
  {
    icon: "📊",
    title: "Clear Verdict Display",
    description: "Get a straightforward verdict: Supported, Refuted, or Inconclusive, based on the evidence.",
  },
  {
    icon: "📚",
    title: "Source Linking",
    description: "Directly access the source research papers (via DOI links) used for the verification.",
  },
  {
    icon: "💡",
    title: "Key Point Summarization",
    description: "Understand the core findings and arguments related to your claim through concise summaries.",
  },
  {
    icon: "✨",
    title: "Interactive Exploration",
    description: "Dive deeper into the analysis, explore related entities, and understand the nuances of the research.",
  },
];

export default function Features() {
  return (
    <div className="features-page container">
      <h1 className="features-title">Discover Factify's Capabilities</h1>
      <p className="features-subtitle">
        Leveraging AI to bring clarity and evidence to scientific claims.
      </p>
      <div className="features-grid">
        {features.map((feature, index) => (
          <div key={index} className="feature-card">
            <div className="feature-icon">{feature.icon}</div>
            <h3 className="feature-card-title">{feature.title}</h3>
            <p className="feature-card-description">{feature.description}</p>
          </div>
        ))}
      </div>
    </div>
  );
} 
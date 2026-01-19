import React from 'react';

const Footer = () => {
  return (
    <footer className="bg-gray-800 text-white p-4 mt-8">
      <div className="container mx-auto text-center">
        <p>&copy; {new Date().getFullYear()} AI Researcher Roadmap. All rights reserved.</p>
        <p className="mt-2">
          Inspired by <a href="https://roadmap.sh" className="text-blue-400 hover:underline">roadmap.sh</a>
        </p>
      </div>
    </footer>
  );
};

export default Footer;

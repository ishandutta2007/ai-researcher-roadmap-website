import React from 'react';

const Header = () => {
  return (
    <header className="bg-gray-800 text-white p-4">
      <div className="container mx-auto flex justify-between items-center">
        <div className="text-2xl font-bold">
          <a href="/">AI Roadmaps</a>
        </div>
        <nav>
          <ul className="flex space-x-4">
            <li><a href="/roadmaps" className="hover:text-gray-400">Roadmaps</a></li>
            <li><a href="/guides" className="hover:text-gray-400">Guides</a></li>
            <li><a href="/about" className="hover:text-gray-400">About</a></li>
          </ul>
        </nav>
      </div>
    </header>
  );
};

export default Header;

import React from 'react';
import Link from 'next/link';

const HomePage = () => {
  const roadmaps = [
    { title: 'Mathematics for AI', slug: 'mathematics-for-ai', description: 'Linear Algebra, Calculus, Probability & Statistics' },
    { title: 'Computer Science Fundamentals', slug: 'cs-fundamentals', description: 'Data Structures, Algorithms, and Programming' },
    { title: 'Core Machine Learning', slug: 'core-ml', description: 'Supervised, Unsupervised, and Reinforcement Learning' },
    { title: 'Deep Learning', slug: 'deep-learning', description: 'Neural Networks, CNNs, RNNs, and Transformers' },
    { title: 'Natural Language Processing (NLP)', slug: 'nlp', description: 'From Text Preprocessing to Large Language Models' },
    { title: 'Computer Vision', slug: 'computer-vision', description: 'Image Processing, Object Detection, and Generation' },
  ];

  return (
    <div>
      <div className="text-center py-16">
        <h1 className="text-5xl font-bold mb-4 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-600">AI Researcher Roadmap</h1>
        <p className="text-xl text-gray-400 mb-8">
          Community driven roadmaps, guides and articles for developers.
        </p>
        <div className="max-w-2xl mx-auto">
          <input
            type="text"
            placeholder="Search for a roadmap..."
            className="w-full p-4 rounded-full bg-gray-800 text-white border-2 border-gray-700 focus:outline-none focus:border-blue-500"
          />
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 mt-16">
        {roadmaps.map((roadmap) => (
          <Link key={roadmap.slug} href={`/roadmaps/${roadmap.slug}`}>
            <div className="bg-gray-800 p-6 rounded-lg text-center h-full hover:bg-gray-700 transition-colors">
              <h2 className="text-2xl font-bold mb-2 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-600">{roadmap.title}</h2>
              <p className="text-gray-400">{roadmap.description}</p>
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
};

export default HomePage;

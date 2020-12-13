import React from 'react';
import logo from './logo.svg';
import ImageUploader from './component/index.js';
import './App.css';

export default class App extends React.PureComponent{
  constructor(props) {
    super(props);
    this.state = {
      title: null
    }
  }

  componentDidMount() {
    fetch('http://localhost:3002/api')
      .then(res => res.json())
      .then(data => this.setState({title: data.title}));
  }

  render() {
    return (
        <div className="page">
            <h1>Socks matching</h1>
            <p>Upload your laundry image.</p>
            <div className="head">
              {this.state.title? <h1>{this.state.title}</h1> : <h1>Upload</h1>}
            </div>
            <ImageUploader style={{ maxWidth: '500px', margin: "20px auto" }}
                           withPreview={true} />
        </div>
    );
}
}


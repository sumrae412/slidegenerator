# Script to Slides Generator

A Flask web application that converts Microsoft Word documents into professional PowerPoint presentations with AI-generated bullet points and visual prompts.

## 🌟 Features

- **Document Conversion**: Convert .docx files to PowerPoint presentations
- **AI-Powered Bullet Points**: Generate concise, complete sentence bullet points using OpenAI GPT
- **Visual Prompt Generation**: Create copyable text prompts for AI image generation
- **Smart Structure Recognition**: Automatically organize content based on heading levels (H1-H4)
- **Table Processing**: Extract content from specific table columns
- **Professional Templates**: Clean, consistent slide layouts
- **Web Interface**: User-friendly drag-and-drop upload interface

## 🚀 Live Demo

Visit the deployed application: [Your Heroku URL will go here]

## 📋 Document Structure Guide

- **H1**: Presentation title page
- **H2**: Section title pages  
- **H3**: Subsection title pages
- **H4**: Individual slide titles
- **Table rows**: Content slides with bullet points

## 🛠️ Installation

### Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd slide_generator
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python wsgi.py
   ```

4. **Access the app**
   Open your browser to `http://localhost:5000`

### Heroku Deployment

1. **Install Heroku CLI**
   ```bash
   # macOS
   brew tap heroku/brew && brew install heroku
   
   # Or download from https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Login to Heroku**
   ```bash
   heroku login
   ```

3. **Create Heroku app**
   ```bash
   heroku create your-app-name
   ```

4. **Deploy**
   ```bash
   git add .
   git commit -m "Initial deployment"
   git push heroku main
   ```

5. **Open your app**
   ```bash
   heroku open
   ```

## 🔧 Configuration

### Environment Variables

The application works without any environment variables, but you can enhance it with:

- **OPENAI_API_KEY** (Optional): For enhanced AI bullet generation
  - Get your key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
  - Without this, basic bullet extraction is used

### File Limits

- **Maximum file size**: 16MB
- **Supported formats**: Microsoft Word (.docx)
- **Processing timeout**: 5 minutes

## 🎯 Usage

1. **Upload Document**: Drag and drop or click to select your .docx file
2. **Select Column**: Choose which table column contains your script text
3. **API Key** (Optional): Enter OpenAI API key for enhanced bullet generation
4. **Convert**: Click "Convert to PowerPoint"
5. **Download**: Get your generated presentation

## 📦 Dependencies

- **Flask 2.3.3**: Web framework
- **python-docx 0.8.11**: Word document processing
- **python-pptx 0.6.21**: PowerPoint generation
- **requests 2.31.0**: HTTP requests for API calls
- **Pillow 10.0.1**: Image processing
- **gunicorn 21.2.0**: Production WSGI server

## 🏗️ Architecture

```
slide_generator/
├── file_to_slides.py      # Main Flask application
├── wsgi.py               # Production entry point
├── templates/            # HTML templates
│   └── file_to_slides.html
├── static/              # CSS, JS, images
├── requirements.txt     # Python dependencies
├── Procfile            # Heroku configuration
├── runtime.txt         # Python version
└── README.md           # This file
```

## 🔒 Privacy & Security

- **No Data Storage**: Files are processed in memory and not stored permanently
- **API Key Security**: OpenAI API keys are used only for the current session
- **File Validation**: Only .docx files under 16MB are accepted
- **Timeout Protection**: Requests automatically timeout after 5 minutes

## 🐛 Troubleshooting

### Common Issues

1. **"Failed to fetch" error**
   - Try without an OpenAI API key first
   - Check file size (must be under 16MB)
   - Ensure file is .docx format

2. **Slow processing**
   - Large files with API keys take longer due to AI processing
   - Consider using without API key for faster processing

3. **Empty bullet points**
   - Check that your document has content in the selected column
   - Verify document structure (tables and headings)

### Getting Help

- Check browser console (F12) for detailed error messages
- Verify file format and size requirements
- Try with a smaller test document first

## 📄 License

This project is available for use under standard terms. See the code for full details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For issues or questions:
- Check the troubleshooting section above
- Review browser console errors
- Test with the provided sample document format
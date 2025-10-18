window.addEventListener('DOMContentLoaded', (event) => {
    // Create the button element
    const button = document.createElement('div');
    button.id = 'floating-chatbot-button';
    button.innerText = 'Chat with AI 🤖';
    
    // Apply styles to the button
    button.style.position = 'fixed';
    button.style.bottom = '20px';
    button.style.right = '20px';
    button.style.backgroundColor = '#4CAF50';
    button.style.color = 'white';
    button.style.padding = '12px 20px';
    button.style.borderRadius = '30px';
    button.style.boxShadow = '0 4px 8px rgba(0,0,0,0.2)';
    button.style.cursor = 'pointer';
    button.style.fontFamily = 'Arial, sans-serif';
    button.style.fontSize = '16px';
    button.style.zIndex = '9999';
    button.style.fontWeight = 'bold';
    button.style.transition = 'all 0.3s ease';
    
    // Add hover effects
    button.addEventListener('mouseenter', function() {
        this.style.backgroundColor = '#3e8e41';
        this.style.boxShadow = '0 6px 12px rgba(0,0,0,0.3)';
        this.style.transform = 'translateY(-2px)';
    });
    
    button.addEventListener('mouseleave', function() {
        this.style.backgroundColor = '#4CAF50';
        this.style.boxShadow = '0 4px 8px rgba(0,0,0,0.2)';
        this.style.transform = 'translateY(0)';
    });
    
    // Add click event
    button.addEventListener('click', function() {
        window.open('https://farmcareaichatbot.vercel.app/', '_blank');
    });
    
    // Add the button to the document body
    document.body.appendChild(button);
    
    // Add responsive styles
    const mediaQuery768 = window.matchMedia('(max-width: 768px)');
    function handleTabletChange(e) {
        if (e.matches) {
            button.style.padding = '10px 16px';
            button.style.fontSize = '14px';
            button.style.bottom = '15px';
            button.style.right = '15px';
        }
    }
    mediaQuery768.addListener(handleTabletChange);
    handleTabletChange(mediaQuery768);
    
    const mediaQuery480 = window.matchMedia('(max-width: 480px)');
    function handleMobileChange(e) {
        if (e.matches) {
            button.style.padding = '8px 12px';
            button.style.fontSize = '12px';
            button.style.bottom = '10px';
            button.style.right = '10px';
        }
    }
    mediaQuery480.addListener(handleMobileChange);
    handleMobileChange(mediaQuery480);
});
### **[] 1. DPM++ 2M Karras**
- **Inference Steps**: 25–30  
- **Guidance Scale (CFG)**: 7.5–8.5  
- **Num Images per Prompt**: 4–6  
- **Why**:  
  - The Karras scheduler’s noise schedule excels at balancing detail and stability, especially with moderate guidance scales.  
  - Steps in this range avoid over-smoothing while capturing fine details from your Textual Inversion embedding.  
  - Use a slightly lower CFG to reduce artifacts while maintaining fidelity to your concept.  

---

### **[] 2. Euler Ancestral**
- **Inference Steps**: 35–40  
- **Guidance Scale (CFG)**: 7–8  
- **Num Images per Prompt**: 6–8  
- **Why**:  
  - Euler Ancestral’s stochasticity helps escape local minima, which is useful for complex or highly unique concepts.  
  - Higher steps compensate for the solver’s randomness, refining details.  
  - Keep CFG moderate to avoid oversaturation of the learned concept.  

---

### **[] 3. DPM++ 2M (Non-Karras)**
- **Inference Steps**: 20–25  
- **Guidance Scale (CFG)**: 8–9  
- **Num Images per Prompt**: 4–6  
- **Why**:  
  - Faster generation with fewer steps, ideal for rapid iteration.  
  - Higher CFG ensures stronger adherence to the Textual Inversion embedding, offsetting the lower step count.  

 


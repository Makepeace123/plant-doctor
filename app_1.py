def display_results(predicted_class, info, confidence):
    plant_type = predicted_class.split('___')[0].replace('_', ' ').title()

    if 'healthy' in predicted_class.lower():
        st.balloons()
        st.success(f"✅ Healthy {plant_type}")
        st.markdown(f"""
        ### Recommendations
        {info['recommendations']}
        
        ### Monitoring Advice
        {''.join([f'- {item}\n' for item in info['monitoring_advice']])}
        """)       
    else:
        disease_name = predicted_class.split('___')[1].replace('_', ' ').title() if '___' in predicted_class else predicted_class.replace('_', ' ').title()
        st.warning(f"⚠️ Detected: {disease_name} ({confidence*100:.1f}% confidence)")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Symptoms", "Prevention", "Treatment", "Chemical Details"])
        
        with tab1:  # Symptoms tab
            st.markdown(f"""
            **Plant Type:** {plant_type}
            
            **Symptoms:**  
            {info['symptoms']}
            
            **Causes:**  
            {info['causes']}
            
            **Effects:**  
            {info['effects']}
            """)
            
        with tab2:  # Prevention tab
            st.markdown("### Prevention Methods")
            st.markdown("#### Cultural Practices")
            for method in info['treatments']['cultural']:
                st.markdown(f"- {method}")  
                
        with tab3:  # Treatment tab
            st.markdown("### Treatment Options")
            
            if info['treatments']['chemical']:
                st.markdown("#### Chemical Treatment")
                chem = info['treatments']['chemical']
                
                st.markdown(f"""
                - **Product:** {chem['product']}
                - **Approx. Price:** {chem.get('price', 'N/A')}  
                - **Dosage:** {chem['dosage']}
                - **Instructions:** {chem.get('note', 'N/A')}
                """)
            else:
                st.info("No chemical treatment recommended")
                
            if info['treatments']['mechanical']:
                st.markdown("#### Mechanical Treatment")
                for method in info['treatments']['mechanical']:
                    st.markdown(f"- {method}")
                
        with tab4:  # Chemical Details tab
            st.info("CAUTION: *Price estimates are approximate and may vary by store/region*")
                
            if info['treatments']['chemical']:
                chem = info['treatments']['chemical']

                st.markdown(f"""
                ### Detailed Chemical Information
                **Product Name:** {chem['product']}  
                **Approx. Market Price:** {chem.get('price', 'Not available')}  
                **Active Ingredient:** {chem.get('active_ingredient', 'N/A')}  
                **Application Frequency:** {chem.get('frequency', 'As needed')}  
                **Safety Precautions:** {chem.get('safety', 'Wear protective gear during application')}
                """)
            else:
                st.info("No chemical treatment details available")

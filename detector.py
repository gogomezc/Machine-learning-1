import cv2
import torch 
#importar yolo 
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

#crear detector
def detector(): 
 
    ##cargar el video, si es una camara la id de la camaraa, puede ser un ip lo que sea
    cap = cv2.VideoCapture("data/streets.mp4")    

#loop que recorre el video frame by frame.
    while cap.isOpened():

        status, frame   = cap.read()
        
        if not status:
            break

        #inferencia
        pred = model(frame)
 
        #xminyminxmaxymax esto analiza el frame y devuelve las coordenadas de los objetos detectados y la confianza de que sea ese objeto o clase
        df = pred.pandas().xyxy[0]
 
        #filtrar por confianza mayor a 0.4 se quedaran como validos, teniendo en cuenta que la confianza va de 0 a 1, y es una estacion de metro por lo que es recurrente el paso de personas
        df = df[df['confidence'] > 0.4]
        
        #recorrer todas las prediciones y dibujar los cuadros y los nombres de las clases     
        for i in range(df.shape[0]):
 
            # le da limites a la box que reconoce el modelo
            bbox = df.iloc[i][['xmin', 'ymin', 'xmax', 'ymax']].values.astype(int)
 
            #print(box)  dibuja el cuadro en la imagen  
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
 
            #print(class_name) escribe el nombre de la clase en la imagen 
            cv2.putText(frame, 
 
                        # pasamos el valor de confianza a porcentaje, y le damos un formato.
                        f"{df.iloc[i]['name']}: {round(df.iloc[i] ['confidence']*100, 2)}%", 
                        (bbox[0], bbox[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 250, 250), 2)
        
        # muestra la imagen en una ventana
        cv2.imshow("frame", frame)

        # si se presiona la tecla q se cierra la ventana
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    #cierra la ventana
    cap.release()
if __name__ == '__main__':
    detector()
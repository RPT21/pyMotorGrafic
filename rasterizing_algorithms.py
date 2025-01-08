import numpy as np
import matplotlib.pyplot as plt
from numba import jit, njit

@njit
def Bresenham_Algorithm(img, x0, y0, x1, y1, color):
    """
    Dibuixa una recta en una imatge representada amb una array.
    
    Paràmetres:
    - img: Array NumPy on es pintarà la recta (modificat in-place).
    - x0, y0: Coordenades del punt inicial.
    - x1, y1: Coordenades del punt final.
    - color: Valor del color (escala de grisos o RGB).
    
    # Algoritme de Bresenham.
    """
    # Calcular diferencies
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 <= x1 else -1
    sy = 1 if y0 <= y1 else -1
    
    # L'error inicial surt d'avançar en X i Y, o sigui, sumar dx i restar dy --- e_xy
    err = dx - dy

    while True:
        # Dibuixar el pixel actual
        img[y0, x0] = color
        
        # Sortir si hem arribat al final
        if x0 == x1 and y0 == y1:
            break
        
        # Calcular error acumulat i ajustar coordenades. Multiplica per 2 per no tenir numeros fraccionaris.
        e2 = 2 * err
        
        # Pendent de la recta: m = (y1-y0)/(x1-x0) = dy/dx
        # Equacio de la recta: [(y-y0)dx - (x-x0)dy = 0] --- recta: y-y0 = m(x-x0)
        # L'error es defineix com la desviacio respecte la recta:
            # err = (y-y0)dx - (x-x0)dy --- Basicament val zero es si estem tocant la recta
            # Si estem per sobre la recta, l'error es positiu, si estem per sota es negatiu
            # Si avancem una unitat en l'eix X, l'error disminueix un dy --- e_y (avancem en l'eix X mantenint Y constant)
            # Si avancem una unitat en l'eix Y, l'error augmenta un dx --- e_x (avancem en l'eix Y mantenint X constant)
            # L'objectiu es sempre fer que l'error s'aproximi el maxim possible a 0.
            # Recordem que sempre dx i dy son nombres positius
            # e_xy es l'error que tenim al avançar en els dos eixos, X i Y. 
            # e_x = e_xy + dy
            # e_y = e_xy - dx
            # e_xy = err + (dx - dy)
        
        if e2 > -dy: # e_xy + e_x > 0 ( err = e_xy )
            # Si avançar en l'eix X i Y dona mes error que nomes avançar en l'eix Y
            x0 += sx # Avancem a l'eix X, ja que minimitza l'error
            err -= dy 
            
        if e2 < dx:  # e_xy + e_y < 0 ( err = e_xy )
            # Si avançar en l'eix X i Y dona mes error que nomes avançar en l'eix X
            y0 += sy # Avancem a l'eix Y, ja que minimitza l'error
            err += dx 
            
    return img


@njit
def PrintTriangleWireframe(img, x0, y0, x1, y1, x2, y2, color):
    Bresenham_Algorithm(img, x0, y0, x1, y1, color)
    Bresenham_Algorithm(img, x0, y0, x2, y2, color)
    Bresenham_Algorithm(img, x1, y1, x2, y2, color)


@njit
def PrintTriangle(img, depth_buffer, texture, x0, y0, x1, y1, x2, y2, color, h0, h1, h2, w0, w1, w2):
    
    # Coordenades de la textura associades a cada vertex numero entre 0 i 1:
    T0X = 0.2 / h0
    T0Y = 0.2 / h0
    
    T1X = 1 / h1
    T1Y = 0.2 / h1
    
    T2X = 0.5 / h2
    T2Y = 1 / h2
    
    texture_height, texture_width = texture.shape[0:2]

    # Suposarem que aquesta funcio s'efectua sobre triangles als quals se'ls ha fet clipping
    # Per evitar que qualsevol dels punts quedi fora de la pantalla
    # Aixo ho fem aixi ja que simplifica molt el calcul de les coordenades baricèntriques
    # Ja que calcular les coordenades baricentriques punt a punt es computacionalment complex
    # Ens basarem en fer increments de les coordenades baricentriques sabent 
    # el punt d'inici i les seves coordenades baricentriques associades
    
    # Guardem les posicions dels vertexs, ja que son importants per determinar les coordenades baricèntriques
    # Vertex A - x0, y0 - coordenada baricentrica: alpha
    # Vertex B - x1, y1 - coordenada baricentrica: beta
    # Vertex C - x2, y2 - coordenada baricentrica: gamma
    
    # Calculem el denominador de les formules de coordenades baricèntriques
    denominador = (y1 - y2)*(x0 - x2) + (x2 - x1)*(y0 - y2)
    
    if denominador == 0:
        # No hi ha area a pintar
        return
        
    # Definim els increments (gradients) de coordenades baricèntriques: alpha, beta, gamma:
    alpha_dx = (y1 - y2) / denominador
    alpha_dy = (x2 - x1) / denominador
    beta_dx = (y2 - y0) / denominador
    beta_dy = (x0 - x2) / denominador
    gamma_dx = (y0 - y1) / denominador
    gamma_dy = (x1 - x0) / denominador

    # Ordenem els vertexs del triangle per altura, ja que pintarem en horitzontal
    # Començarem pintant des del vertex de menor altura, i anirem pintant cada segment horitzontal
    # Calculem les coordenades baricentriques inicials - comencem sempre al vertex A
    
    Ax = x0
    Ay = y0
    Bx = x1
    By = y1
    
    if y1 < y0: # Intercanviem vertex A amb vertex B
        x0, x1 = x1, x0
        y0, y1 = y1, y0
        
    if y2 < y0: # Intercanviem vertex A amb vertex C
        x0, x2 = x2, x0
        y0, y2 = y2, y0
        
    if y2 < y1: # Intercanviem vertex B amb vertex C
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        
    if x0 == Ax and y0 == Ay:
        alpha = 1 
        beta = 0
        gamma = 0
    elif x0 == Bx and y0 == By:
        alpha = 0
        beta = 1
        gamma = 0 
    else:
        alpha = 0
        beta = 0
        gamma = 1
    
    # Calcular diferencies
    dx_012 = abs(x1 - x0)
    dy_012 = abs(y1 - y0)
    sx_012 = 1 if x0 <= x1 else -1
    sy_012 = 1
    
    dx_02 = abs(x2 - x0)
    dy_02 = abs(y2 - y0)
    sx_02 = 1 if x0 <= x2 else -1
    sy_02 = 1
    
    # Calculem els errors inicials:
    err_012 = dx_012 - dy_012
    err_02 = dx_02 - dy_02
    
    # Definim les variables que anirem modificant
    x0_012 = x0
    y0_012 = y0
    
    x0_02 = x0
    y0_02 = y0
    
    # Pintarem les linies horitzontals començant des del costat llarg "02"
    # Anem a determinar el sentit amb el que pintarem les linies horitzontals:
    # Definim la recta que va de (x0, y0) -> (x2, y2), i calculem el punt de interseccio amb la recta y = y1
    # Equacio recta: y - y0 = m(x - x0) -> on m = (y2 - y0) / (x2 - x0) -> aillant la x:
    x_intersect = ((x2 - x0) / (y2 - y0)) * (y1 - y0) + x0
    
    # Si el punt de interseccio esta a la dreta de x1, pintem de dreta a esquerra
    # Si el punt de interseccio esta a la esquerra de x1, pintem de esquerra a dreta
    sx = -1 if (x_intersect - x1) > 0 else 1
    
    while True:
        
        n = 0
        for x in range(x0_02, x0_012 + sx, sx):
            
            # El pixel buffer tindra el invers de les distancies, anira de 0 a infinit
            # Com mes a prop estigui un objecte, mes gran sera el valor del invers:
            
            # Calculem les coordenades baricentriques del triangle projectat
            alpha_2D = alpha + n * sx * alpha_dx
            beta_2D = beta + n * sx * beta_dx
            gamma_2D = gamma + n * sx * gamma_dx
                
            # Interpolacio amb perspectiva amb les coordenades baricentriques -> Z-buffer  
            pixel_depth_inv = alpha_2D * (1 / h0) + beta_2D * (1 / h1) + gamma_2D * (1 / h2)
            
            # Coordenades de la textura interpolades amb perspectiva:
            u = (alpha_2D * T0X + beta_2D * T1X + gamma_2D * T2X) / pixel_depth_inv
            v = (alpha_2D * T0Y + beta_2D * T1Y + gamma_2D * T2Y) / pixel_depth_inv
            tex_x = int(u * texture_width)
            tex_y = int(v * texture_height)
            
            # Assegurem que les coordenades estiguin dins els limits:
            tex_x = max(0, min(texture_width - 1, tex_x))
            tex_y = max(0, min(texture_height - 1, tex_y))
            
            if depth_buffer[y0_02, x] < pixel_depth_inv:
                
                depth_buffer[y0_02, x] = pixel_depth_inv
                
                # Pintem el pixel corresponent
                img[y0_02, x] = texture[tex_y, tex_x]
                # img[y0_02, x] = color
            
            n += 1
        
        while True:
            
            # Aquesta recta es la recta curta, que te 2 segments, i s'ha de
            # recalcular quan arriba al final del primer
    
            # Canviem de costat si hem arribat al final
            if x0_012 == x1 and y0_012 == y1:
                dx_012 = abs(x2 - x1)
                dy_012 = abs(y2 - y1)
                sx_012 = 1 if x1 <= x2 else -1
                err_012 = dx_012 - dy_012
            
            e2_012 = 2 * err_012
            
            if e2_012 > -dy_012:
                x0_012 += sx_012
                err_012 -= dy_012 
                
            if e2_012 < dx_012:
                y0_012 += sy_012
                err_012 += dx_012
                break
        
        while True:
            
            # Aquesta es la recta llarga, la que va del punt mes baix al punt mes alt
            
            e2_02 = 2 * err_02
            
            if e2_02 > -dy_02:
                
                x0_02 += sx_02
                err_02 -= dy_02 
                
                alpha += alpha_dx * sx_02
                beta += beta_dx * sx_02
                gamma += gamma_dx * sx_02
                
                
            if e2_02 < dx_02:
                
                y0_02 += sy_02
                err_02 += dx_02
                
                alpha += alpha_dy
                beta += beta_dy
                gamma += gamma_dy
                
                break
            
        if y0_02 == y2:
            
            # Si hem arribat al final, pintem l'ultim punt i sortim del loop:
                
            # Interpolacio amb perspectiva amb les coordenades baricentriques -> Z-buffer  
            pixel_depth_inv = alpha * (1 / h0) + beta * (1 / h1) + gamma * (1 / h2)
            
            # Coordenades de la textura interpolades amb perspectiva:
            u = (alpha * T0X + beta * T1X + gamma * T2X) / pixel_depth_inv
            v = (alpha * T0Y + beta * T1Y + gamma * T2Y) / pixel_depth_inv
            tex_x = int(u * texture_width)
            tex_y = int(v * texture_height)
            
            # Assegurem que les coordenades estiguin dins els limits:
            tex_x = max(0, min(texture_width - 1, tex_x))
            tex_y = max(0, min(texture_height - 1, tex_y))
            
            if depth_buffer[y0_02, x0_02] < pixel_depth_inv:
                
                depth_buffer[y0_02, x0_02] = pixel_depth_inv
               
                # Pintem el pixel corresponent
                # img[y0_02, x0_02] = color
                img[y0_02, x0_02] = texture[tex_y, tex_x]
                
            break


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
def PrintTriangle(img, depth_buffer, texture, x0, y0, x1, y1, x2, y2, color, 
                  h0, h1, h2, T0X, T0Y, T1X, T1Y, T2X, T2Y, changeBase_light_matrix, 
                  projectedA, projectedB, projectedC, vertex_normals, d, 
                  phongShading = False, useTexture = False, bilinearFilter = False):

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
    
    if useTexture:
        # Coordenades de la textura associades a cada vertex numero entre 0 i 1:
        T0X = T0X / h0
        T0Y = T0Y / h0
        
        T1X = T1X / h1
        T1Y = T1Y / h1
        
        T2X = T2X / h2
        T2Y = T2Y / h2
        
    if phongShading:
        # Guardem les normals dels vertexs per despres interpolar amb perspectiva:
        normalA = vertex_normals[0] / h0
        normalB = vertex_normals[1] / h1
        normalC = vertex_normals[2] / h2
    
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
    
    # Considerem una variable boolena per fer que la comprovacio de canvi de segment
    # curt nomes es faci quan sigui necessari:
    LineChanged = False
    
    # Pintarem les linies horitzontals començant des del costat llarg "02"
    # Anem a determinar el sentit amb el que pintarem les linies horitzontals:
    # Definim la recta que va de (x0, y0) -> (x2, y2), i calculem el punt de interseccio amb la recta y = y1
    # Equacio recta: y - y0 = m(x - x0) -> on m = (y2 - y0) / (x2 - x0) -> aillant la x:
    x_intersect = ((x2 - x0) / (y2 - y0)) * (y1 - y0) + x0
    
    # Si el punt de interseccio esta a la dreta de x1, pintem de dreta a esquerra
    # Si el punt de interseccio esta a la esquerra de x1, pintem de esquerra a dreta
    sx = -1 if (x_intersect - x1) > 0 else 1
    
    while True:
        
        # Guardem la X just abans d'avançar. Si avancem en el sentit contrari
        # al de pintar, aquesta X es la que ens importa
        x_final = x0_012
        
        while True:
            
            # Aquesta recta es la recta curta, que te 2 segments, i s'ha de
            # recalcular quan arriba al final del primer
    
            # Canviem de costat si hem arribat al final de la primera linea
            # La comprovacio nomes s'ha de fer si encara no hem canviat de linea
            if not LineChanged:
                if x0_012 == x1 and y0_012 == y1:
                    dx_012 = abs(x2 - x1)
                    dy_012 = abs(y2 - y1)
                    sx_012 = 1 if x1 <= x2 else -1
                    err_012 = dx_012 - dy_012
                    LineChanged = True
                    
            e2_012 = 2 * err_012

            if e2_012 > -dy_012:
                x0_012 += sx_012
                err_012 -= dy_012
                
                # Guardem la X nomes si esta avançant en el sentit de pintar
                # i si la Y no avança:
                if sx == sx_012 and not e2_012 < dx_012:
                    x_final = x0_012
                    continue
            
            if e2_012 < dx_012:
                y0_012 += sy_012
                err_012 += dx_012
                break
            
            if y0_012 == y2:
                break
         
        # Un cop la linea curta ha incrementat la seva y en una unitat, sabem
        # l'interval de x que hem de pintar, ja que tenim guardada la x que 
        # delimita el segment horitzontal (x_final)
        
        n = 0
        for x in range(x0_02, x_final + sx, sx):
            
            # El pixel buffer tindra el invers de les distancies, anira de 0 a infinit
            # Com mes a prop estigui un objecte, mes gran sera el valor del invers:
            
            # Calculem les coordenades baricentriques del triangle projectat
            alpha_2D = alpha + n * sx * alpha_dx
            beta_2D = beta + n * sx * beta_dx
            gamma_2D = gamma + n * sx * gamma_dx
                
            # Interpolacio amb perspectiva amb les coordenades baricentriques -> Z-buffer  
            pixel_depth_inv = alpha_2D * (1 / h0) + beta_2D * (1 / h1) + gamma_2D * (1 / h2)
            
            n += 1
            
            if depth_buffer[y0_02, x] < pixel_depth_inv:
                
                depth_buffer[y0_02, x] = pixel_depth_inv
                
                if useTexture:
                    
                    texture_height, texture_width = texture.shape[0:2]
                    # Coordenades de la textura interpolades amb perspectiva:
                    u = (alpha_2D * T0X + beta_2D * T1X + gamma_2D * T2X) / pixel_depth_inv
                    v = (alpha_2D * T0Y + beta_2D * T1Y + gamma_2D * T2Y) / pixel_depth_inv
                    
                    tex_x = u * texture_width
                    tex_y = v * texture_height
                    
                    # Assegurem que les coordenades estiguin dins els limits:
                    tex_x = max(0, min(texture_width - 1, tex_x))
                    tex_y = max(0, min(texture_height - 1, tex_y))
                    
                    if bilinearFilter:
                    
                        tx = int(tex_x)
                        ty = int(tex_y)
                        fx = tex_x - tx
                        fy = tex_y - ty
                        
                        TL = texture[ty, tx]
                        
                        if tx + 1 < texture_width:
                            TR = texture[ty, tx + 1]
                        else:
                            TR = texture[ty, tx]
                            
                        if ty + 1 < texture_height:
                            BL = texture[ty + 1, tx]
                        else:
                            BL = texture[ty, tx]
                            
                        if tx + 1 < texture_width and ty + 1 < texture_height:
                            BR = texture[ty + 1, tx + 1]
                        else:
                            BR = texture[ty, tx]
                            
                        CT = fx * TR + (1 - fx) * TL
                        CB = fx * BR + (1 - fx) * BL
                        
                        texture_color = fy * CB + (1 - fy) * CT
                        
                    else:
                        
                        tx = int(tex_x)
                        ty = int(tex_y)
                        texture_color = texture[ty, tx].astype(np.float64)
                
                if phongShading:
                    
                    # Calculem el phong Shading:
                    projected_point = alpha_2D * projectedA + beta_2D * projectedB + gamma_2D * projectedC
                    real_point = projected_point / (d * pixel_depth_inv)
                    real_point[2] = 1 / pixel_depth_inv
                    
                    # Interpolem la normal a cada pixel amb perspectiva:
                    vertex_normal = (alpha_2D * normalA + beta_2D * normalB + gamma_2D * normalC) / pixel_depth_inv
                    
                    # Analitzem quina cara del triangle estem mirant, recordem que la camera esta al origen:
                    camera_vector = -real_point
                    camera_vector_norm = np.linalg.norm(camera_vector)
                    camera_vector = camera_vector / camera_vector_norm
                    dot_prod_camera_triangle = np.dot(camera_vector, vertex_normal)
                    if dot_prod_camera_triangle > 0:
                        isTriangleFrontside = True
                    else:
                        isTriangleFrontside = False
                    
                    # Calculem la llum que arriba a cada pixel del triangle:
                    pixel_intensity = 0
                    for light in changeBase_light_matrix:
                        if light[3] == 0 or light[3] == 1: # LLum puntual o direccional
                            
                            if light[3] == 0:
                                L = light[0:3] - real_point
                            else:
                                L = light[0:3]
                                
                            L_norm = np.linalg.norm(L)
                            L = L / L_norm
                            
                            dot_prod_light = np.dot(L, vertex_normal)
                            
                            if dot_prod_light > 0:
                                isLightFrontside = True
                            else:
                                isLightFrontside = False
                                
                            if isTriangleFrontside == isLightFrontside:
                                pixel_intensity += light[4] * abs(dot_prod_light)
                        else:
                            pixel_intensity += light[4]
                            
                    if pixel_intensity > 1:
                        pixel_intensity = 1
                
                else:
                    pixel_intensity = 1
                
                # Pintem el pixel corresponent
                if useTexture:
                    img[y0_02, x] = texture_color * pixel_intensity
                else:
                    img[y0_02, x] = color * pixel_intensity
        
        if y0_02 == y2:
            # Si hem pintat la ultima linea, sortim del programa:
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
                
                if sx_02 == sx:
                    # Si estem avançant en el mateix sentit que el de pintar,
                    # segur que estem al borde del triangle
                    break
                
            if y0_02 == y2:
                # Si hem arribat a l'ultim punt:
                break
            
            # Si estem avançant en sentit contrari al de pintar:
            if sx_02 != sx:
                
                # Mirem l'accio posterior que fara la recta
                e2_02 = 2 * err_02
                
                # Si la X avança pero la Y no avança:
                if e2_02 > -dy_02 and not e2_02 < dx_02:
                    # Podem avançar
                    continue
                else:
                    # Aturem el programa si la Y avança
                    break



(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a d)
(on b e)
(ontable c)
(ontable d)
(on e a)
(ontable f)
(clear b)
(clear c)
(clear f)
)
(:goal
(and
(on b e)
(on d f)
(on f c))
)
)


